import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import random
import pandas as pd


def main():
    # Check device compatibility
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Paths to datasets
    train_pos_path = "twitter-datasets/train_pos_full.txt"
    train_neg_path = "twitter-datasets/train_neg_full.txt"
    test_data_path = "twitter-datasets/test_data.txt"

    # Load datasets with shuffling
    def load_data(pos_path, neg_path):
        # Read positive tweets
        with open(pos_path, "r") as f:
            positive_tweets = f.readlines()
        # Read negative tweets
        with open(neg_path, "r") as f:
            negative_tweets = f.readlines()

        # Combine tweets and labels
        texts = positive_tweets + negative_tweets
        labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

        # Shuffle the data
        combined = list(zip(texts, labels))  # Pair tweets with labels
        random.shuffle(combined)  # Shuffle the pairs
        texts, labels = zip(*combined)  # Unzip back into texts and labels
        return list(texts), list(labels)

    train_texts, train_labels = load_data(train_pos_path, train_neg_path)

    # Split training data into train and eval subsets
    train_texts_split, eval_texts, train_labels_split, eval_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42
    )

    # Load test dataset
    with open(test_data_path, "r") as f:
        test_texts = f.readlines()

    # Dataset class
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
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            item = {key: val.squeeze(0) for key, val in inputs.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    # Use BERT-tiny for testing purposes
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
    model.to(device)

    # Create datasets
    train_dataset = SentimentDataset(train_texts_split, train_labels_split, tokenizer)
    eval_dataset = SentimentDataset(eval_texts, eval_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, tokenizer=tokenizer)

    # Evaluation metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Perform evaluation after every epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        use_cpu=device.type == "cpu",  # Use CPU if MPS is unavailable
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Include evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Make predictions on the test set
    predictions = trainer.predict(test_dataset)
    predicted_labels = predictions.predictions.argmax(-1)

    # Save predictions in the required format
    output = pd.DataFrame({
        "Id": range(1, len(predicted_labels) + 1),  # Generate sequential IDs starting from 1
        "Prediction": predicted_labels
    })
    output.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()
