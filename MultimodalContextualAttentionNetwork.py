import torch
import torch.nn as nn
from ContextualTransformer import ContextualTransformer

class MultimodalContextualAttention(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(MultimodalContextualAttention, self).__init__()

        self.text_transformer = ContextualTransformer(d_model, nhead, dim_feedforward, dropout)
        self.visual_transformer = ContextualTransformer(d_model, nhead, dim_feedforward, dropout)

        # Initialize an alpha parameter
        self.alpha_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, text, visual):

        C_TI = self.text_transformer(text, visual)

        C_IT = self.visual_transformer(visual, text)

        alpha = torch.sigmoid(self.alpha_param)
        beta = 1 - alpha

        return alpha*C_TI + beta*C_IT