from dataset import MultiTaskDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = MultiTaskDataset(tokenizer, split="train")

sample = dataset[0]
print(sample)
