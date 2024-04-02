import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50


class HierarchicalEncodingNetwork(nn.Module):
    def __init__(self, mcan_model, text_d_model, img_d_model, bert_output_layers=12, groups=3) -> None:
        super().__init__(HierarchicalEncodingNetwork, self)

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        
        # Adapt ResNet for feature extraction to match the image embedding size
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, img_d_model)
        
        # Multimodal Contextual Attention Network model
        self.mcan = mcan_model
        
        # Number of groups to split BERT layers
        self.layer_groups = groups

        # Total number of BERT output layers
        self.bert_output_layers = bert_output_layers


    def forward(self, input_ids, attention_mask, images):

        # Extract hidden states from BERT
        hidden_states = self.bert(input_ids, attention_mask=attention_mask).hidden_states
        
        # Process image through ResNet
        img_features = self.resnet(images)
        
        # Initialize a list to hold MCAN outputs for each group
        mcan_outputs = []

        # Group BERT layers' outputs and process each group with the image features through MCAN
        layers_per_group = self.bert_output_layers // self.layer_groups

        for i in range(self.layer_groups):
            # Average the outputs of the layers in the current group
            group_layers = hidden_states[i * layers_per_group : (i + 1) * layers_per_group]

            group_avg = torch.mean(torch.stack(group_layers), dim=0)
            
            # Get the [CLS] token representation as the text feature
            text_features = group_avg[:, 0, :]
            
            # Process through MCAN and collect the output
            mcan_output = self.mcan(text_features, img_features)
            mcan_outputs.append(mcan_output)
        
        # Concatenate all MCAN outputs to form the final multimodal representation
        combined_output = torch.cat(mcan_outputs, dim=-1)
        
        return combined_output