import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os


def load_data(pos_path, neg_path):
    def read_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.readlines()

    positive_tweets = read_file(pos_path)
    negative_tweets = read_file(neg_path)

    texts = positive_tweets + negative_tweets
    labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding=False,  # Let `DataCollatorWithPadding` handle padding dynamically
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    # Check device compatibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths to datasets
    train_pos_path = "train_pos.txt"
    train_neg_path = "train_neg.txt"
    test_data_path = "test_data.txt"

    # Load and prepare datasets
    train_texts, train_labels = load_data(train_pos_path, train_neg_path)
    train_texts_split, eval_texts, train_labels_split, eval_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    with open(test_data_path, "r", encoding="utf-8") as f:
        test_texts = f.readlines()

    # Initialize tokenizer and model
    model_name = "textattack/bert-base-uncased-imdb"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Create datasets
    train_dataset = SentimentDataset(train_texts_split, train_labels_split, tokenizer)
    eval_dataset = SentimentDataset(eval_texts, eval_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, tokenizer=tokenizer)

    # Data collator for efficient batching
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # Slightly higher learning rate for fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
        num_train_epochs=4,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=0.1,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Fine-tune the model
    trainer.train()

    # Make predictions on the test set
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    # Format predictions for output
    formatted_predictions = pd.DataFrame({
        "Id": range(1, len(predicted_labels) + 1),
        "Prediction": [1 if x == 1 else -1 for x in predicted_labels]
    })

    # Save predictions
    formatted_predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
