import torch 
from torch import nn
from model.core_transformer_block import CoreTransformerEncoder
from config import EnvConfig, GeneralConfig
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert, FlowExpert


class FlowsheetNetwork(nn.Module):
    
    def __init__(self, gen_config, env_config, device: torch.device = None):
        super().__init__()
        
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = gen_config.latent_dim
        self.num_heads = gen_config.num_heads
        self.num_blocks = gen_config.num_transformer_blocks

        # First build up Transformer Encoder using blocks 
        self.core_transformer = nn.ModuleList([])
        for _ in range(self.num_blocks):
            block = CoreTransformerEncoder(d_model = self.latent_dim, nhead = self.num_heads, 
                                           dropout=gen_config.dropout, 
                                           clip_value = 10)
            self.core_transformer.append(block)

        self.virtual_node_embedding = nn.Embedding(num_embeddings = 4, embedding_dim=self.latent_dim) # decisions across 4 lvls 

        #----heads----#
        self.terminate_head = nn.Linear(self.latent_dim, 1)
        self.open_stream_head = nn.Linear(self.latent_dim, 1)
        
        # define all unit experts to get embeddings 
        self.unit_experts = {
            "distillation_column": DistillationColumn(gen_config), 
            "decanter": Decanter(gen_config),   
            "mixer": Mixer(gen_config),
            "split": Split(gen_config),
            "recycle": Recycler(gen_config), 
            "add_solvent": AddSolvent(gen_config, env_config),    
            "flow_expert": FlowExpert(gen_config, env_config),
            }
        
        self.edge_expert = EdgeFlowExpert(gen_config, self.unit_experts["flow_expert"])
        self.open_stream_expert = OpenStreamExpert(gen_config, env_config, self.unit_experts["flow_expert"])


    def forward(self, x):
        
        # All nodes and edges already in latent space and batched per flowsheet (N = all nodes within the graph, d= latent_dim, K= num of open stream)
        batch_latent_nodes_embed = x["batch_latent_nodes"] # (B, N, d) 
        batch_latent_edges_embed = x["batch_latent_edges"] # (B, N + 1, N + 1, d)
        state_info = x["state_information"]
        open_stream_masks = x["open_stream_mask"]
        levels = torch.tensor([s['current_level'] for s in state_info], dtype=torch.long)
        valid_nodes = x["node_valid_mask"] # for padding for additive attention

        # Create attentive masks for transformer 
        padding_attn_mask = (~valid_nodes[:, :, None] | ~valid_nodes[:, None, :]) # (B, N, N)
        padding_attn_mask = padding_attn_mask.float() * (-1e9)
    
        level_embed = self.virtual_node_embedding(levels).unsqueeze(1) # only virtual residue gets level info
        batch_latent_nodes_embed = torch.cat([level_embed, batch_latent_nodes_embed], dim=1) # (B, N + 1, d)

        # Send node and edge embeddings to core transformer 
        for block in self.core_transformer:
            nodes_out = block(batch_latent_nodes_embed, batch_latent_edges_embed, padding_attn_mask)

        # lvl 0: make predictions whether to terminate or open stream
        latent_virtual_node = nodes_out[:, 0, :] # (B, d)
        terminate_logits = self.terminate_head(latent_virtual_node) # (B,) 

        # Now extract node embedding corresponding to a specific node id for open streams 
        latent_nodes_transformed = nodes_out[:, 1:, :] # (B, N, d) 
        batch_idx = torch.arange(latent_nodes_transformed.size(0), device=latent_nodes_transformed.device).unsqueeze(1)
        selected_node_logits = latent_nodes_transformed[batch_idx, x["batch_open_node_indices"]]  # (B, 1, d)

        # Get embedding corresponding to outlets 
        outlet_embeds = self.open_stream_expert.outlet_emb(x["batch_open_outlet_indices"])
        stream_embeds = selected_node_logits + outlet_embeds

        # Now provide stream embedding to open_stream head to get logits 
        open_stream_logits = self.open_stream_head(stream_embeds).squeeze(-1)
        open_stream_logits = open_stream_logits.masked_fill(~x["open_stream_mask"], -1e9) # mask out all non valid streams from padding

        terminate_or_open_stream_logits = torch.cat([terminate_logits, open_stream_logits], dim=1)  # (B, K+1)
        
        # lvl 1: if open stream, logits for predictions for units 
        unit_predictions = {}
        for unit_type, expert in self.unit_experts.items():
            if unit_type not in ["mixer", "recycle", "flow_expert"]:
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed)
            elif unit_type == "recycle":
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed, x['recycler_masks'])
            elif unit_type == "mixer":
                unit_predictions[unit_type] = expert.predict(latent_nodes_transformed, x['mixer_masks'])
            
        return terminate_or_open_stream_logits, unit_predictions, open_stream_masks, valid_nodes
    
    
    def get_weights(self):
        return dict_to_cpu(self.state_dict())
    
    
def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


