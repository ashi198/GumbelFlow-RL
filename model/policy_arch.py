import torch 
from torch import nn
from model.core_transformer_block import CoreTransformerEncoder
from config import EnvConfig, GeneralConfig
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert


class FlowsheetNetwork(nn.Module):
    
    def __init__(self, device: torch.device = None):
        super().__init__()
        
        self.device = torch.device("cpu") if device is None else device
        gen_config = GeneralConfig()
        self.latent_dim = gen_config.latent_dim
        self.num_heads = gen_config.num_heads
        self.num_blocks = gen_config.num_transformer_blocks

        # First build up Transformer Encoder using blocks 
        self.core_transformer = nn.ModuleList([])
        for _ in range(self.num_blocks):
            block = CoreTransformerEncoder(d_model = self.latent_dim, nhead = self.num_heads, dropout=gen_config.dropout, mask = None, clip_value = 10)
            self.core_transformer.append(block)

        self.virtual_node_embedding = nn.Embedding(num_embeddings= 4, embedding_dim=self.latent_dim) # decisions across 4 lvls 

        #----heads----#
        self.logit_terminate_or_pick_stream = nn.Linear(self.latent_dim, 1)
        self.per_unit_head = nn.Linear(self.latent_dim, 6) 
        
        # define all unit experts to get embeddings 
        self.unit_experts = {
            "distillation_column": DistillationColumn(gen_config), 
            "decanter": Decanter(gen_config),   
            "mixer": Mixer(gen_config),
            "split": Split(gen_config),
            "recycle": Recycler(gen_config), 
            "add_solvent": AddSolvent(gen_config),    
            "open_stream": OpenStreamExpert(gen_config)     
            }
        
        self.edge_expert = EdgeFlowExpert()

    def forward(self, x):
        
        # all nodes and edges already in latent space and batched per flowsheet
        batch_latent_nodes_embed = x["batch_latent_nodes"] # (B, N+1, d)
        batch_latent_edges_embed = x["batch_latent_edges"] # (B, N+1, N+1, d)
        state_info = x["current_state"]

        batch_latent_nodes_embed[:, -1, :] = batch_latent_nodes_embed[:, -1, :] + self.virtual_node_embedding(state_info.level) # only virtual residue gets level info

        # Send node and edge embeddings to core transformer 
        nodes_out = self.core_transformer(batch_latent_nodes_embed, batch_latent_edges_embed)

        # lvl 0: make predictions whether to terminate or open stream

        latent_virtual_node = nodes_out[:, -1, :] # (B, d)
        latent_nodes_transformed = nodes_out[:, 0:-1, :] # (B, N, d)

        # level 0 logits
        lvl0_input = torch.cat([latent_nodes_transformed, latent_virtual_node.unsqueeze(1)], dim=1)  # (B, N+1, d)

        terminate_or_open_stream_logits = self.logit_terminate_or_pick_stream(lvl0_input).squeeze(-1)
        
        # lvl 1: if open stream, logits for predictions for units 
        unit_predictions = {}
        for unit_type, expert in self.unit_experts.items():
            unit_predictions[unit_type] = expert.predict(latent_nodes_transformed)

        # implement masking logic here 
        return terminate_or_open_stream_logits, unit_predictions
    
    
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

