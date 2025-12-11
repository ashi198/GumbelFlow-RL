import torch 
from torch import nn
from core_transformer_block import CoreTransformerEncoder
from experts.experts import DistillationColumn, Decanter, Mixer, Split, Recycler, AddSolvent, EdgeFlowExpert, OpenStreamExpert

class FlowsheetNetwork(nn.Module):
    
    def __init__(self, config, device: torch.device = None):
        super(FlowsheetNetwork).__init__()
        
        self.device = torch.device("cpu") if device is None else device
        config = self.config
        self.latent_dim = config.latent_dim
        self.num_heads = config.num_heads
        self.num_blocks = config.num_transformer_blocks


        # First build up Transformer Encoder using blocks 
        self.core_transformer = nn.ModuleList([])
        for _ in range(self.num_blocks):
            block = CoreTransformerEncoder(d_model = self.latent_dim, nhead = self.num_heads, dropout=config.dropout, mask = None, clip_value = 10)
            self.core_transformer.append(block)

        # Define all experts here 
        self.unit_experts = dict(
            Distillation = DistillationColumn(), 
            Decanter = Decanter(),   
            Mixer = Mixer(),
            Split = Split(),
            Recycler = Recycler(), 
            AddSolvent = AddSolvent(),    
            OpenStream = OpenStreamExpert()     

         )
        
        self.edge_expert = EdgeFlowExpert()

        self.logit_terminate_or_pick_stream = nn.Linear(self.latent_dim, 1)

    def forward(self, flowsheet):

        # Embed all nodes within the flowsheet into the latent space using experts 

        latent_nodes_embed = [self.unit_experts[node.type].embed(node) for node in flowsheet]
        latent_edge_embed = [self.edge_expert(edge) for edge in flowsheet]

        # Send node and edge embeddings to core transformer 
        latent_nodes_transformed, latent_virtual_node = self.core_transformer(latent_nodes_embed, latent_edge_embed)

        # lvl 0: make predictions whether to terminate or open stream
        terminate_or_open_stream_logits = self.logit_terminate_or_pick_stream(
            torch.cat(latent_virtual_node, latent_nodes_transformed)
            )
        
        # implement masking logic here 
        
        # lvl 1: if open stream, logits for predictions for units 
        unit_predictions = dict ()
        for unit_type, expert in self.unit_experts:
            unit_predictions[unit_type] = dict(
            )

        # implement masking logic here 

        return terminate_or_open_stream_logits, unit_predictions



