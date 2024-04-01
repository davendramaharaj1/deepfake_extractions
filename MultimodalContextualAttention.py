import torch.nn as nn
from ContextualTransformer import ContextualTransformer

class MultimodalContextualAttention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(MultimodalContextualAttention, self).__init__()

        self.text_transformer = ContextualTransformer(d_model, nhead, dim_feedforward, dropout)
        self.visual_transformer = ContextualTransformer(d_model, nhead, dim_feedforward, dropout)

    def forward(self, text, visual):

        C_TI = self.text_transformer(text, visual)

        C_IT = self.visual_transformer(visual, text)

        return C_TI, C_IT