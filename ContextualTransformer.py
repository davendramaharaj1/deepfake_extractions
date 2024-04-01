import torch
import torch.nn as nn
from CrossAttentionLayer import CrossAttentionLayer
from torch.nn import TransformerEncoderLayer, AdaptiveAvgPool1d

class ContextualTransformer(nn.Module):
    '''
        Contextual Transformer comprises 2 parts:
        1. Standard Encoder Layer
        2. Cross Attention layer

        Encoder Layer uses self attention since it receives one input
        Cross Attention layer receives two inputs via two different modalities (text and input)
    '''
    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(ContextualTransformer, self).__init__()

        # Encoder with Self Attention Layer receiving input1
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        # Cross Attention Layer
        self.cross_attn = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        self.pooling1 = AdaptiveAvgPool1d(1)
        self.pooling2 = AdaptiveAvgPool1d(1)

    def forward(self, input1, input2):

        # pass input 1 through encoder
        f_single = self.encoder_layer(input1)

        # pool the output of encoder
        f_single_pooled = self.pooling1(f_single)

        # pass f_single and input2 through Cross attention layer
        f_co = self.cross_attn(f_single.unsqueeze(1), input2, input2)

        # pool f_co
        f_co_pooled = self.pooling2(f_co)

        out = torch.concat((f_single_pooled, f_co_pooled), dim=1)

        return out