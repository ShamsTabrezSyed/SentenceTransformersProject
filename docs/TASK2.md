# Task 2: Multi-Task Learning (MTL) Expansion

## 1. Architecture
The model is based on the pre-trained `distilbert-base-uncased` transformer. We added:
- **Classification Head**: A linear layer applied to the `[CLS]` token.
- **NER Head**: A linear layer applied to the output embeddings of all tokens for NER.

### Model Architecture Overview:
1. **Input Layer**: Tokenized input text.
2. **Transformer Encoder**: A shared transformer encoder (`distilbert-base-uncased`).
3. **Classification Head**: A linear layer to predict a class label based on the `[CLS]` token.
4. **NER Head**: A linear layer applied to all token embeddings to predict token-wise labels for NER.

## 2. Training Strategy
The training process alternates between the classification and NER tasks within a batch. The training loop operates as follows:
- **Classification Task**: For each batch, a sample of text is passed through the model's classification head, and a loss is computed using **CrossEntropyLoss**.
- **NER Task**: A sample of text is passed through the model's NER head, and a loss is computed for each token using **CrossEntropyLoss** (with padding tokens ignored using `ignore_index=-100`).

The total loss for each batch is the sum of the classification and NER losses.

### Loss Functions:
- **Classification Loss**: Cross-entropy loss for the classification head.
- **NER Loss**: Cross-entropy loss for the NER head (with `ignore_index=-100` for padding tokens).

## 3. Data Preparation
The dataset is in **JSON format**, with each sample containing:
- `sentence`: The input text.
- `label`: The class label for classification.
- `ner_tags`: A list of named entity tags for the sentence.

### Example of a `train.json` record:
```json
{
    "sentence": "John lives in New York.",
    "label": 1, 
    "ner_tags": ["B-PER", "O", "O", "B-LOC"]
}
NER tags:

"B-PER" indicates the beginning of a person entity.
"O" indicates non-entity tokens.
"B-LOC" indicates the beginning of a location entity.

4. Model Training
The model is trained for multiple epochs. In each epoch, the model processes batches of data from both the classification and NER tasks. The losses from both tasks are computed separately and then summed up for backpropagation.

Optimizer: AdamW

Epochs: 2

Checkpointing: Model saved every epoch.

5. Model Evaluation (Optional)
After training, you can evaluate the model on a separate validation dataset. A typical evaluation step would involve running the trained model on the validation set, calculating the classification and NER metrics (accuracy, F1-score, etc.).

6. Inferences
The trained model can be used for making inferences on unseen text. A predict.py script is available for making predictions using the saved model.

Example Prediction:
Using the trained model for classification:

bash
Copy
Edit
$ python src/predict.py --model_path saved_model/mtl_epoch1.pt --text "John lives in New York." --task classification
This will output:

vbnet
Copy
Edit
Task: classification
Input: John lives in New York.
Prediction: 1
Example Prediction for NER:
Using the trained model for NER:

bash
Copy
Edit
$ python src/predict.py --model_path saved_model/mtl_epoch1.pt --text "John lives in New York." --task ner
This will output:

pgsql
Copy
Edit
Task: ner
Input: John lives in New York.
NER Output: ['B-PER', 'O', 'O', 'B-LOC']
##7. Key Points
Two-task model: Classification and NER handled by a shared transformer encoder.

Alternating tasks: During training, the model alternates between processing classification and NER samples.

Separate losses: Each task has its own loss function (CrossEntropyLoss), and the losses are summed up during backpropagation.

8. Future Improvements
Freezing Layers: For transfer learning, freezing the transformer layers and fine-tuning only the task-specific heads can be considered.

Multi-Class Classification: If more than two classes are needed for the classification task, adjust the output size of the classification head.

Fine-Tuning on More Data: Adding more data for both tasks will improve model performance.

sql
Copy
Edit

### Steps to Add and Commit:
Once you have updated the `task2.md` file with the above content, add and commit it to your repository:

```bash
git add docs/task2.md
git commit -m "Added Task 2 Multi-Task Learning documentation"
git push origin task-2-multitask-training








