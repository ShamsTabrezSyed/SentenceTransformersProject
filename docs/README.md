# Multi-Task Learning with Sentence Transformers

## Overview
This project implements a multi-task learning (MTL) architecture using a transformer backbone (`distilbert-base-uncased`) to perform both:

1. **Sentence Classification** – Assign a class label to an entire sentence.
2. **Named Entity Recognition (NER)** – Tag each token in a sentence with an entity label.

It builds on a pre-trained sentence transformer for embedding extraction, adds two task-specific heads, and trains them jointly via an alternating mini‑batch strategy.

## Project Structure
```
SentenceTransformersProject/
│
├── requirements.txt          # Python dependencies               
│
├── src/                      # Source code
│   ├── dataset.py            # MultiTaskDataset loader
│   ├── multitask_model.py    # Model: shared encoder + two heads
│   ├── train.py              # Training loop with checkpointing
│   ├── evaluate.py           # Evaluation on validation set
│   ├── config.yaml           # Hyperparameters & paths
│   ├── predict.py            # Inference script for new sentences
│   └──data/                   # JSON datasets
│        ├── train.json            # Training samples
│        └── valid.json            # Validation samples
│
├── saved_model/              # Saved checkpoints after training
├── docs/                     # Documentation and sample outputs
│    ├── sample_output.txt    # Example inference outputs
│    ├── README.md            # Project overview (this file)
│
│
└── docker/                   # Docker setup (optional)
    ├── Dockerfile
    └── docker-compose.yml
```

## Installation
Ensure you have Python 3.7+ and pip installed. Then:
```bash
pip install -r requirements.txt
```  
This installs:
- torch
- transformers
- sentence-transformers
- pyyaml
- scikit-learn
- datasets
- tqdm

## Configuration
Edit `config.yaml` to customize:
```yaml
model_name: distilbert-base-uncased
max_length: 128
batch_size: 16
learning_rate: 2e-5
num_epochs: 3
num_classes: 2
num_ner_labels: 5
train_file: data/train.json
valid_file: data/valid.json
save_dir: saved_model
```  
Paths and hyperparameters can be changed as needed.

## Training
Run the training script, specifying epochs, learning rate, and checkpoint directory:
```bash
python src/train.py --epochs 3 --lr 3e-5 --checkpoint_dir saved_model
```  
- Model checkpoints (`mtl_epoch<epoch>.pt`) will be saved under `saved_model/`.

## Evaluation
Evaluate on the validation set and view classification & NER metrics:
```bash
python src/evaluate.py
```  
This loads the latest checkpoint and prints precision, recall, and F1 for each task.

## Inference
Run predictions on new sentences:
```bash
python src/predict.py
```  
Example output:
```
Classification: 1  # class index
NER Tags: [("John", 1), ("lives", 0), ...]
```

## Task Breakdown
### Task 1: Sentence Transformer Implementation
- Used `sentence-transformers/all-MiniLM-L6-v2` backbone.
- Encoded input sentences into fixed-size (384-d) embeddings.
- Chose dropout + linear projection for classification head.

### Task 2: Multi-Task Learning Expansion
- **Architecture:** Shared transformer + two heads:
  - Classification head on [CLS] token.
  - NER head on token embeddings.
- **Training:** Alternating mini‑batches between tasks; separate loss functions.
- **Data Handling:** Padded/truncated NER labels to sequence length with `ignore_index=-100`.

### Task 3: Training Considerations
- **Freezing Entire Network:** When embeddings suffice—only heads train.
- **Freezing Backbone Only:** Adapt heads quickly without altering pre-trained features.
- **Freezing One Head:** Fine-tune one task while holding the other stable.
- **Transfer Learning:** Select a strong pre-trained model (e.g., BERT), freeze its lower layers, unfreeze higher layers and heads.

### Task 4: Training Loop Implementation
- **Assumptions:** Synthetic data loader; batch_size=1 for clarity.
- **Forward Pass:** Route per-task through appropriate head.
- **Metrics:** Classification accuracy; NER token-level accuracy.

  

