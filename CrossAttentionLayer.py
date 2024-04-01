import torch.nn as nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ReLU

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(CrossAttentionLayer, self).__init__()

        # MultiHead Attention component
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feedforward Network component
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Normalization Layers
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Activation function
        self.activation = ReLU

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        '''
        Unlike standard Encoder Structure, this layer receives input from two different sources
        for HMCAN, input2 = query, f_single = key = value
        '''
        # Multihead Attention
        attn_output = self.multihead_attn(query, key, value, 
                                          attn_mask=None, 
                                          key_padding_mask=key_padding_mask,
                                          need_weights=need_weights)
        
        # Add + Norm Layer after multihead attention 
        query = self.norm1(query + self.dropout(attn_output))

        # Feedforward Network
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(query))))

        # Second Add + Norm Layer after feeedforward network
        ff_out = self.norm2(query + self.dropout(ff_out))

        return ff_out