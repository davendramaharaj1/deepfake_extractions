import torch
import torch.nn as nn
from CrossAttentionLayer import CrossAttentionLayer
from torch.nn import TransformerEncoderLayer, AdaptiveAvgPool1d
import torch.nn.functional as F

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

    # def forward(self, input1, input2):

    #     print('Inside Contextual transformer')
    #     print(f'Input1: {input1.shape}')
    #     print(f'Input2: {input2.shape}')
    #     print('Entering encoder...\n')
    #     # pass input 1 through encoder
    #     f_single = self.encoder_layer(input1)
        
    #     print(f'After encoder, output: {f_single.shape}')
    #     # pool the output of encoder
    #     print('Entering Cross Attention...\n')

    #     # pass f_single and input2 through Cross attention layer
    #     f_co = self.cross_attn(input2, f_single, f_single)

    #     print(f'size of f_co: {f_co.shape}')

    #     print('Pooling...')

    #     # For f_single_pooled, where the output is likely [batch, seq_len, features]
    #     f_single_pooled = self.pooling1(f_single.transpose(1, 2))  # you might need to transpose dimensions
    #     f_single_pooled = f_single_pooled.view(f_single_pooled.size(0), -1)  # Flatten to [batch, features]

    #     # Similar adjustment for f_co_pooled
    #     f_co_pooled = self.pooling2(f_co.transpose(1, 2))  # Ensure the sequence length is treated as 'L' in AdaptiveAvgPool1d
    #     f_co_pooled = f_co_pooled.view(f_co_pooled.size(0), -1)
    #     f_co_pooled = f_co_pooled.view(1, -1)  # This reshapes it to [1, 49*768]

    #     # f_single_pooled = self.pooling(f÷_single)
    #     print(f'pooled fsingle: {f_single_pooled.shape}')

    #      # pool f_co
    #     # f_co_pooled = self.pooling2(f_co)
    #     print(f'size of pooled f_co: {f_co_pooled.shape}')

    #     out = torch.concat((f_single_pooled, f_co_pooled), dim=1)

    #     return out

    def forward(self, input1, input2):

        print(f'Input1 before encoding: {input1.shape}')
        print(f'Input2 before cross attn: {input2.shape}')

        input1_encoded = self.encoder_layer(input1)

        # Padding input2 to match the sequence length of input1
        if input2.size(1) < input1.size(1):
            padding_size = input1.size(1) - input2.size(1)
            input2 = F.pad(input2, (0, 0, 0, padding_size))

        # Create a mask for input2
        input2_pad_mask = input2.sum(dim=-1) == 0  # Padding mask

        cross_attn_output = self.cross_attn(input2, input1_encoded, input1_encoded, key_padding_mask=input2_pad_mask).permute(1, 0, 2)

        print(f'Input1 after encoding: {input1_encoded.shape}')
        print(f'Input2 after cross attn: {cross_attn_output.shape}')

        # Pooling
        print('Pooling outputs...')
        input1_pooled = self.pooling1(input1_encoded.transpose(1, 2)).squeeze(-1)
        cross_attn_pooled = self.pooling2(cross_attn_output.transpose(1, 2)).squeeze(-1)

        print(f'Input1 after pooling: {input1_pooled.shape}')
        print(f'Input2 after pooling: {cross_attn_pooled.shape}')

        # Concatenate pooled outputs
        combined_output = torch.cat([input1_pooled, cross_attn_pooled], dim=-1)
        print(f'Combined output: {combined_output.shape}')

        return combined_output