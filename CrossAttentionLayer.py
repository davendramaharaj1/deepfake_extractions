import torch.nn as nn
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
import torch.nn.Functional as F

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
        print('Inside Cross Attention')
        print(f'Size of Query: {query.shape}')
        print(f'Size of Key: {key.shape}')
        print(f'Size of Value: {value.shape}\n')

        # key = key.squeeze()
        # value = value.squeeze()
        print('Re-arranging Q, K, V')
        query = query.transpose(0, 1)  # From [1, 49, 768] to [49, 1, 768]
        key = key.transpose(0, 1)      # From [1, 512, 768] to [512, 1, 768]
        value = value.transpose(0, 1)  # From [1, 512, 768] to [512, 1, 768]
        print(f'Size of Query: {query.shape}')
        print(f'Size of Key: {key.shape}')
        print(f'Size of Value: {value.shape}\n')


        # print('After squeezing KEY AND VALUE')
        # print(f'Size of Query: {query.shape}')
        # print(f'Size of Key: {key.shape}')
        # print(f'Size of Value: {value.shape}')
        # print('Entering MultiHead\n')

        # Multihead Attention
        attn_output, _ = self.multihead_attn(query, key, value, 
                                          attn_mask=None, 
                                          key_padding_mask=key_padding_mask,
                                          need_weights=need_weights)
        
        # Add + Norm Layer after multihead attention 
        query = self.norm1(query + self.dropout(attn_output))

        print('After multi head')
        print(f'Size of Query: {query.shape}')

        # Feedforward Network
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(query))))

        # Second Add + Norm Layer after feeedforward network
        ff_out = self.norm2(query + self.dropout(ff_out))

        return ff_out