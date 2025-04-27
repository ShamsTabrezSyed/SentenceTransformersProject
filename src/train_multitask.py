import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import argparse
import os

# Dummy dataset loader
from dataset import MultiTaskDataset  # Ensure this file exists in src/

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


def main(epochs, lr, checkpoint_dir):
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = MultiTaskDataset(tokenizer, split="train")
    val_dataset = MultiTaskDataset(tokenizer, split="val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Initialize model, optimizer, criteria
    model = MultiTaskModel("distilbert-base-uncased", num_cls_labels=2, num_ner_labels=5)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)

    # Ensure checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_cls_loss, total_ner_loss = 0.0, 0.0

        for batch in train_loader:
            task = batch["task"][0]
            text = batch["text"][0]
            print("Training on:", task)

            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            optimizer.zero_grad()

            if task == "classification":
                label = torch.tensor([batch["label"][0]], dtype=torch.long)
                print("Classification label:", label.item())
                logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
                loss = criterion_cls(logits, label)
                total_cls_loss += loss.item()

            elif task == "ner":
                # Convert list of tensors to python list of ints
                original_tensor_list = batch["ner_labels"]  # list of tensors
                original_labels = [int(x.item()) for x in original_tensor_list]
                print("NER labels:", original_labels)

                seq_len = input_ids.size(1)
                # Pad or truncate labels to match seq_len, using ignore_index=-100 for padding
                padded = original_labels + [-100] * max(0, seq_len - len(original_labels))
                padded = padded[:seq_len]
                labels = torch.tensor([padded], dtype=torch.long)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
                loss = criterion_ner(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_ner_loss += loss.item()

            else:
                continue

            # Backward and optimize
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}/{epochs} - Classification Loss: {total_cls_loss:.4f}, NER Loss: {total_ner_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"mtl_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # (Optional) validation step can be added here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr, checkpoint_dir=args.checkpoint_dir)
