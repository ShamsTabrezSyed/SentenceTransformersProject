import torch
import torch.nn as nn  # Ensure you have this import to define the model
from transformers import AutoModel

class MultiTaskModel(nn.Module):
    def __init__(self, base_model_name, num_cls_labels, num_ner_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_cls_labels)
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        if task == "classification":
            cls_output = sequence_output[:, 0, :]
            return self.classifier(cls_output)
        elif task == "ner":
            return self.ner_head(sequence_output)
        else:
            raise ValueError(f"Unknown task: {task}")
