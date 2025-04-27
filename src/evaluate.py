import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from multitask_model import MultiTaskModel
from dataset import MultiTaskDataset

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def evaluate(model, dataloader, device):
    model.eval()
    all_classification_preds = []
    all_classification_labels = []

    all_ner_preds = []
    all_ner_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            classification_labels = batch['classification_label'].to(device)
            ner_labels = batch['ner_labels'].to(device)

            classification_logits, ner_logits = model(input_ids, attention_mask)

            classification_preds = torch.argmax(classification_logits, dim=1)
            ner_preds = torch.argmax(ner_logits, dim=2)

            all_classification_preds.extend(classification_preds.cpu().numpy())
            all_classification_labels.extend(classification_labels.cpu().numpy())

            # For NER, flatten the sequences
            for true_seq, pred_seq in zip(ner_labels, ner_preds):
                true_seq = true_seq.cpu().numpy()
                pred_seq = pred_seq.cpu().numpy()
                
                # Mask out padding (-100 usually for ignored labels)
                mask = true_seq != -100
                all_ner_labels.extend(true_seq[mask])
                all_ner_preds.extend(pred_seq[mask])

    return all_classification_preds, all_classification_labels, all_ner_preds, all_ner_labels

def main():
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load validation dataset
    valid_dataset = MultiTaskDataset(config['valid_data_path'], config)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

    # Load model
    model = MultiTaskModel(config)
    model.load_state_dict(torch.load(config['saved_model_path'], map_location=device))
    model = model.to(device)

    # Evaluate
    cls_preds, cls_labels, ner_preds, ner_labels = evaluate(model, valid_loader, device)

    # Classification report
    print("\nClassification Task Report:")
    print(classification_report(cls_labels, cls_preds))

    # NER report
    print("\nNER Task Report:")
    print(classification_report(ner_labels, ner_preds))

if __name__ == "__main__":
    main()
