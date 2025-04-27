from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.split = split
        if split == "train":
            self.data = [
                {"text": "John lives in New York", "task": "ner", "ner_labels": [1, 0, 0, 2, 3]},
                {"text": "This is a positive example", "task": "classification", "label": 1},
            ]
        else:
            self.data = [
                {"text": "Mary works at Google", "task": "ner", "ner_labels": [1, 0, 0, 2]},
                {"text": "This is a negative example", "task": "classification", "label": 0},
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
