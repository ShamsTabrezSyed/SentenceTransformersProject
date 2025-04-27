import torch
from transformers import AutoTokenizer
from multitask_model import MultiTaskModel  # Ensure this is correct and points to your model file
import argparse

def load_model(model_path, base_model_name, num_classes, num_ner_tags):
    print(f"Loading model from {model_path}")
    model = MultiTaskModel(base_model_name, num_classes, num_ner_tags)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.eval()
    return model

def predict(model, tokenizer, text, task):
    print(f"Predicting task: {task} on text: {text}")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, task=task)
    
    if task == "classification":
        print(f"Classification prediction: {output.argmax(dim=1).item()}")
        return output.argmax(dim=1).item()
    elif task == "ner":
        # You can add logic for NER here if needed
        print(f"NER prediction: {output}")
        return output
    else:
        print("Unknown task.")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--text", type=str, required=True, help="Text to predict on")
    parser.add_argument("--task", type=str, choices=["classification", "ner"], required=True, help="Task type")
    parser.add_argument("--base_model_name", type=str, default="distilbert-base-uncased", help="Base model name")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classification labels")
    parser.add_argument("--num_ner_tags", type=int, default=5, help="Number of NER tags")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, args.base_model_name, args.num_classes, args.num_ner_tags)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    # Make prediction
    result = predict(model, tokenizer, args.text, args.task)
    
    if result is not None:
        print(f"Prediction result: {result}")

if __name__ == "__main__":
    main()
