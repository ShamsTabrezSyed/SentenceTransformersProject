import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, base_model='sentence-transformers/all-MiniLM-L6-v2', num_classes=3, num_ner_tags=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)

        # Sentence Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, num_classes)
        )

        # Named Entity Recognition Head
        self.ner_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, num_ner_tags)
        )

    def forward(self, input_ids, attention_mask, task="classification"):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if task == "classification":
            cls_embedding = outputs.last_hidden_state[:, 0]
            return self.classifier(cls_embedding)
        elif task == "ner":
            token_embeddings = outputs.last_hidden_state
            return self.ner_head(token_embeddings)
        else:
            raise ValueError("Unsupported task: choose either 'classification' or 'ner'")
