# train_multitask.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from multitask_model import MultiTaskSentenceTransformer  # assuming your class is in multitask_model.py

# Hypothetical dataset
data = [
    {"sentence": "I love this!", "label": 1, "task": "classification"},
    {"sentence": "Paris is the capital of France.", "ner_labels": [0, 1, 0, 0, 2, 0], "task": "ner"}
]

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = MultiTaskSentenceTransformer()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion_cls = nn.CrossEntropyLoss()
criterion_ner = nn.CrossEntropyLoss()

model.train()

for epoch in range(2):
    total_cls_loss = 0
    total_ner_loss = 0
    for sample in data:
        inputs = tokenizer(sample["sentence"], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if sample["task"] == "classification":
            label = torch.tensor([sample["label"]])
            logits = model(input_ids=input_ids, attention_mask=attention_mask, task="classification")
            loss = criterion_cls(logits, label)
            total_cls_loss += loss.item()
        elif sample["task"] == "ner":
            labels = torch.tensor([sample["ner_labels"]])
            logits = model(input_ids=input_ids, attention_mask=attention_mask, task="ner")
            loss = criterion_ner(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_ner_loss += loss.item()
        else:
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Classification Loss = {total_cls_loss:.4f}, NER Loss = {total_ner_loss:.4f}")
