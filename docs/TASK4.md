# Task 4: Training Loop Implementation (BONUS)

This section details the complete training loop used for the multi-task model (classification + NER). It highlights assumptions, data handling, the forward pass, loss computation, backpropagation, and metrics logging.

---

## 1. Assumptions & Setup

- **Data Format**: Each sample is a dict with keys:
  - `"text"` (string)
  - `"task"` ("classification" or "ner")
  - `"label"` (int) for classification or
  - `"ner_labels"` (list of ints) for NER

- **Batch Size**: We use `batch_size=1` for clarity (alternating single samples). In practice, you can increase this.

- **Tokenizer & Model**:
  ```python
  tokenizer = AutoTokenizer.from_pretrained(base_model)
  model = MultiTaskSentenceTransformer(
      base_model=base_model,
      num_classes=num_cls_labels,
      num_ner_tags=num_ner_labels
  )
  ```

- **Loss Functions**:
  ```python
  criterion_cls = nn.CrossEntropyLoss()
  criterion_ner = nn.CrossEntropyLoss(ignore_index=-100)
  ```

- **Optimizer**: AdamW with learning rate `lr`, e.g., `2e-5`.

---

## 2. High-Level Training Loop

```python
for epoch in range(1, epochs + 1):
    model.train()
    total_cls_loss = 0.0
    total_ner_loss = 0.0

    for batch in train_loader:  # batch_size=1
        task = batch['task'][0]
        text = batch['text'][0]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        optimizer.zero_grad()

        if task == 'classification':
            label = torch.tensor([batch['label'][0]], dtype=torch.long)
            logits = model(input_ids, attention_mask, task)
            loss = criterion_cls(logits, label)
            total_cls_loss += loss.item()

        elif task == 'ner':
            # Pad/truncate ner_labels to sequence length
            seq_len = input_ids.size(1)
            original = batch['ner_labels'][0]
            padded = original + [-100] * max(0, seq_len - len(original))
            padded = padded[:seq_len]
            labels = torch.tensor([padded], dtype=torch.long)

            logits = model(input_ids, attention_mask, task)
            loss = criterion_ner(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_ner_loss += loss.item()

        loss.backward()
        optimizer.step()

    # End of epoch losses
    print(f"Epoch {epoch}/{epochs} - Classification Loss: {total_cls_loss:.4f}, NER Loss: {total_ner_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"{checkpoint_dir}/mtl_epoch{epoch}.pt")
```

---

## 3. Forward Pass Details

- **Classification**:
  1. Extract `[CLS]` token embedding: `sequence_output[:, 0, :]`.
  2. Pass through `classifier` head: `Dropout → Linear`.
  3. Compute softmax logits and loss.

- **NER**:
  1. Use all token embeddings: `sequence_output` shape `(1, seq_len, hidden_size)`.
  2. Pass each token embedding through `ner_head`: `Dropout → Linear`.
  3. Reshape logits to `(1*seq_len, num_tags)` and labels to `(1*seq_len,)` for loss.

---

## 4. Metrics & Logging

- **Loss Tracking**: Sum classification and NER losses separately each epoch.
- **Optional Accuracy Computation**:
  ```python
  # Classification accuracy
  preds = logits.argmax(dim=1)
  cls_acc = (preds == label).float().mean()

  # NER token accuracy (ignore padding)
  token_preds = logits.view(-1, num_ner_tags).argmax(dim=-1)
  mask = labels.view(-1) != -100
  ner_acc = (token_preds[mask] == labels.view(-1)[mask]).float().mean()
  ```
- **Example Log**:
  ```
  Epoch 1/3 - Classification Loss: 0.98, NER Loss: 1.58
  Epoch 2/3 - Classification Loss: 0.45, NER Loss: 1.32
  Epoch 3/3 - Classification Loss: 0.22, NER Loss: 1.05
  ```

---

## 5. Key Decisions

- **Batch Size = 1**: Simplifies alternating tasks in a highly interpretable way (could be batched in practice).
- **`ignore_index=-100` for NER**: Prevents padded tokens from contributing to loss.
- **Separate Loss Computation**: Allows monitoring each task’s learning progress independently.

## 6. Summary
This training loop demonstrates how to integrate an alternating, per-task training regime for a multi-task transformer model. It handles tokenization, padded label alignment, forward/backward passes, and separate loss tracking, providing a clear blueprint for extending to larger datasets and batched inputs.

