import os
import json
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# Model and Dataset Initialization
# Purpose: Load the model, tokenizer, and dataset for fine-tuning and evaluation.
def initialize_model_and_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)  # Adjust for your use case
    dataset = load_dataset("your_dataset")  # Replace 'your_dataset' with actual dataset
    return model, tokenizer, dataset

# Data Preprocessing
# Purpose: Tokenize dataset and prepare it for model input.
def preprocess_data(tokenizer, dataset):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# Training Function
# Purpose: Fine-tune the BERT model.
# Input: Preprocessed dataset, model, and training parameters.
# Output: Trained model.
def train_model(model, tokenized_datasets):
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    trainer.train()
    return model

# Evaluation Function
# Purpose: Evaluate the fine-tuned model.
# Input: Model and evaluation dataset.
# Output: Evaluation metrics.
def evaluate_model(model, tokenized_datasets):
    from transformers import Trainer

    trainer = Trainer(model=model)
    results = trainer.evaluate(tokenized_datasets["test"])
    print("Evaluation Results:", results)
    return results

# Main Execution Pipeline
if __name__ == "__main__":
    # Initialize model and dataset
    model, tokenizer, dataset = initialize_model_and_data()

    # Preprocess data
    tokenized_datasets = preprocess_data(tokenizer, dataset)

    # Train model
    model = train_model(model, tokenized_datasets)

    # Evaluate model
    metrics = evaluate_model(model, tokenized_datasets)

    # Save metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Pipeline completed successfully.")
