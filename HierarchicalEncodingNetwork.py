import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50

class HierarchicalEncodingNetwork(nn.Module):
    def __init__(self, mcan_model, bert, resnet, output_dim, groups=3) -> None:
        super().__init__(HierarchicalEncodingNetwork, self)

        self.bert = bert
        self.resnet = resnet
        
        # Multimodal Contextual Attention Network model
        self.mcan = mcan_model
        
        # Number of groups to split BERT layers
        self.layer_groups = groups

        self.output_dim = output_dim

        self.classifier = nn.Linear(output_dim*groups, 2)


    def forward(self, input_ids, attention_mask, images):

        # Extract hidden states from BERT
        hidden_states = self.bert(input_ids, attention_mask=attention_mask).hidden_states
        
        # Process image through ResNet
        img_features = self.resnet(images)

        # Group BERT layers' outputs and process each group with the image features through MCAN
        grouped_text_features = self._group_bert_outputs(hidden_states)

        combined_outputs = []
        
        for text_features in grouped_text_features:
            mcan_output = self.mcan(text_features, img_features)
            combined_outputs.append(mcan_output)

        final_features = torch.cat(combined_outputs, dim=-1)
        predictions = self.classifier(final_features)
        return predictions
    
    def _group_bert_outputs(self, hidden_states):
        # Simplified grouping strategy
        step = len(hidden_states) // self.groups

        return [torch.mean(torch.stack(hidden_states[i*step:(i+1)*step]), dim=0) for i in range(self.groups)]