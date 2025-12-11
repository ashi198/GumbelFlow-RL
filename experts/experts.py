from torch import nn
import torch 
from model.core_transformer_block import CoreTransformerEncoder 


# Implementation of all unit experts
# make sure to add recylcer mask in batch_process 
# normalization for AddSolvent components? 
# normalization for interaction components in Flow Expert? 

class DistillationColumn(nn.Module):
    def __init__(self, config):
        super(DistillationColumn, self).__init__()
        self.name = "distillation_column"
        self.config = config 
        self.latent_dim = config.latent_dim
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.df_categorical = nn.Linear(in_features = self.latent_dim, out_features = 100)
        self.df_and_type_embed = nn.Embedding(in_features = 1, out_features =  self.latent_dim)

    def embed(self, df: torch.FloatTensor):
        return self.df_and_type_embed(df)
        
    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch, num_nodes, latent_dim)
        
        return {
            "picked_logit": self.logit_linear(x), 
            "distillate_fraction_categorical": self.df_categorical(x)
            }
    

class Decanter(nn.Module):
    def __init__(self, config):
        super(Decanter, self).__init__()
        self.name = "decanter"
        self.config = config 
        self.latent_dim = config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Embedding(in_features = 1, out_features = self.latent_dim)

    def embed(self, batch_size: int):
        return self.type_embed(
            torch.tensor([0] * batch_size, dtype = torch.long) #(batch_size, latent_dim)
        )
        
    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch, num_nodes, latent_dim)
        
        return {
            "picked_logit": self.logit_linear(x), 
            }
    
class Mixer(nn.Module):
    def __init__(self, config):
        super(Decanter, self).__init__()
        self.name = "mixer"
        self.config = config 
        self.latent_dim = config.latent_dim
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Embedding(in_features = 1, out_features = self.latent_dim)

    def embed(self, batch_size: int):
        return self.type_embed(
            torch.tensor([0] * batch_size, dtype = torch.long) #(batch_size, latent_dim)
        )
        
    def predict(self, x: torch.FloatTensor):
        # x is the embedding of shape (batch, num_nodes, latent_dim)

        return {
            "picked_logit": self.logit_linear(x), 
            }
    

class Split(nn.Module):
    def __init__(self, config):
        super(Split, self).__init__()
        self.name = "split"
        self.config = config 
        self.latent_dim = config.latent_dim
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.split_ratio_categorical = nn.Linear(in_features = self.latent_dim, out_features = 100)
        self.split_ratio_and_type_embed = nn.Embedding(in_features = 1, out_features = self.latent_dim)

    def embed(self, sr: torch.FloatTensor):
        return self.split_ratio_and_type_embed(sr)
        
    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch, num_nodes, latent_dim)
        
        return {
            "picked_logit": self.logit_linear(x), 
            "split_ratio_categorical": self.split_ratio_categorical(x)
            }
    
class Recycler(nn.Module):
    def __init__(self, config, recycler_mask: torch.Tensor):
        super(Recycler, self).__init__()
        self.name = "recycler"
        self.config = config 
        self.recycler_mask = recycler_mask 
        self.latent_dim = config.latent_dim 
        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.query_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.key_linear_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.scale_constant = 10

    def predict(self, x: torch.FloatTensor):

        # x is the embedding of shape (batch_size, num_nodes, latent_dim)

        query = self.query_linear_proj(x)
        key = self.key_linear_proj(x)
        scores = torch.einsum('bnd, bmd -> bnm', query, key)
        scores = self.scale_constant * torch.tanh(scores)

        # mask out not allowed nodes (e.g self connections, system source nodes)
        scores = scores.masked_fill(self.recycler_mask == 0, float('-inf')) 

        return {
        "picked_logit": self.logit_linear(x), # (batch_size, num_nodes, 1)
        "target_scores": scores # (batch_size, num_nodes, num_nodes)
        } 
    
class AddSolvent(nn.Module):
    def __init__(self, config):
        super(AddSolvent, self).__init__()
        self.name = "add_solvent"
        self.config = config 
        self.latent_dim = config.latent_dim 

        self.logit_linear = nn.Linear(in_features = self.latent_dim, out_features = 1)
        self.type_embed = nn.Linear(in_features = 4, out_features = self.latent_dim)

        self.prediction_mlp = nn.Sequential(
            nn.Linear(self.latent_dim + 4, 2 * self.latent_dim),
            nn.SiLU(),
            nn.Linear(2 * self.latent_dim, 1 + 100) # for logit and amount discretized to 100 categories
        )

    def embed(self, x):
        return self.type_embed(x)

    def predict(self, x: torch.FloatTensor, components: torch.FloatTensor):

        # x is the embedding of shape (batch_size, num_nodes, latent_dim)
        # components is of shape (num_possible_components, 4)
        batch_size, num_nodes, _ = x.size()
        num_components, _ = components.size()

        components = components[None, ...].repeat(num_nodes, 1, 1)
        components = components[None, ...].repeat(batch_size, 1, 1, 1) # (batch_size, num_nodes, num_possible_components, 3)

        mlp_in = x[:, :, None, :].repeat(1, 1, num_components, 1)
        mlp_in = torch.cat([mlp_in, components], dim = -1) #(batch_size, num_nodes, num_componets, 4 + latent_dim)
        mlp_out = self.prediction_mlp(mlp_in) #(batch_size, num_nodes, num_components, 1 + 100 )

        return {
        "picked_logit": self.logit_linear(x), # (batch_size, num_nodes, 1)
        "component_logit": mlp_out[:, :, :, 0].sequeeze(-1), # (batch_size. num_nodes, num_components) <- distribution over which component to add 
        "component_amount": mlp_out[:, :, :, 1:] # (batch_size, num_nodes, num_components, 100)
        } 
    
class FlowExpert(nn.Module):
    def __init__(self,config):
        super(FlowExpert, self).__init__()
        self.config = config 
        self.flow_latent_dim = config.latent_dim / 8
        self.num_trf_blocks = config.num_trf_flow_blocks

        self.flow_trf_encoder = nn.ModuleList([])
        for _ in range(self.num_trf_blocks):
            block = CoreTransformerEncoder(d_model = self.flow_latent_dim, nhead = 4, dropout = config.dropout,
                                          )
            self.flow_trf_encoder.append(block)

        self.component_linear = nn.Linear(in_features = 3, out_features = self.flow_latent_dim)
        self.edge_linear = nn.Linear(in_features = 1, out_features = self.flow_latent_dim)
        self.flow_latent_upscale = nn.Linear (in_features = self.flow_latent_dim, out_features = config.latent_dim)

    def forward(self, component_params, interaction_params, amount):

            # component_params: (batch_size, num_components, 3)
            # interaction_params: (batch_size, num_components, num_components, 1)
            # amount: (batch_size, num_components, 1)
        
        nodes = self.component_linear(component_params) * amount # (batch_size, num_components, flow_latent_dim)
        edges = self.edge_linear(interaction_params) # (batch_size, num_components, num_components, flow_latent_dim)
        transformed_nodes = self.flow_trf_encoder(nodes, edges) # (batch_size, num_components, num_components, flow_latent_dim)
        flow_embedding = transformed_nodes.mean(dim=1) # (batch_size, flow_latent_dim)
        return self.flow_latent_upscale(flow_embedding) # (batch_size, latent_dim)

class EdgeFlowExpert(nn.Module):
    def __init__(self, config, flow_expert: FlowExpert):
        super(EdgeFlowExpert, self).__init__()
        self.flow_expert = flow_expert 
        self.latent_dim = config.latent_dim

        # no edge connection embedding 
        self.is_recycle_emb = nn.Embedding(in_features = 2, out_features = self.latent_dim) # 0 for no, 1 for yes

    def forward(self, x):
        latent_flow = self.flow_expert(x) # check how to make this as component_params, interactions_params etc
        recycle_emb = self.is_recycle_emb()
        combine_edge_embed = latent_flow + recycle_emb 

        return combine_edge_embed
    

class OpenStreamExpert(nn.Module):
    def __init__(self, config, flow_expert: FlowExpert):
        super(OpenStreamExpert, self).__init__()
        self.flow_expert = flow_expert 
        self.latent_dim = config.latent_dim
        self.linear_transform_open_stream = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        latent_flow = self.flow_expert(x) # check how to make this as component_params, interactions_params etc
        open_stream_embed = self.linear_transform_open_stream(latent_flow)
        
        return open_stream_embed



