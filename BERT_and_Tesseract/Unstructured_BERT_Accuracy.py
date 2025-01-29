import json
import torch
import numpy as np
import re
import random
from collections import defaultdict, Counter
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load extracted data
with open('<BASE_PATH>/extracted_text_data.json', 'r') as f:
    text_data = json.load(f)

# Randomly sample
# Purpose: Limit dataset size by sampling a maximum of 2000 entries.
# Input: Full dataset.
# Output: Subsampled dataset containing at most 2000 entries.
random.seed(42)
sample_image_files = random.sample(list(text_data.keys()), min(2000, len(text_data)))
text_data = {key: text_data[key] for key in sample_image_files}

# Define labels
unique_labels = ["paragraph", "page_header", "subheading", "title", "table", "footer", "header"]
data = []

# Extract data into usable format
for image_name, content in text_data.items():
    for label in unique_labels:
        if label in content and content[label]:
            data.append({"text": ' '.join(content[label]), "label": label})

# Check label distribution
label_counts = Counter(item['label'] for item in data)
print("\nLabel Distribution:")
for label in unique_labels:
    print(f"{label}: {label_counts.get(label, 0)}")

# Validate data availability
if not data:
    raise ValueError("No valid data with 'text' and 'label' found.")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Initialize tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(unique_labels)).to(device)

# Tokenize dataset
def tokenize(batch):
    """
    Purpose: Tokenize input text.
    Input: batch (dict): Dictionary containing text data.
    Output: Tokenized batch (dict).
    """
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)

# Encode labels
# Purpose: Convert string labels into integer indices for the model.
# Input: Dataset with string labels.
# Output: Dataset with encoded labels.
label_encoder = ClassLabel(names=unique_labels)
dataset = dataset.map(lambda example: {"label": label_encoder.str2int(example["label"])})

# Split dataset
# Purpose: Divide the dataset into training, validation, and test sets.
# Input: Full dataset.
# Output: Training, validation, and test datasets.
train_valid, test_dataset = dataset.train_test_split(test_size=0.15, seed=42).values()
train_dataset, validation_dataset = train_valid.train_test_split(test_size=0.1765, seed=42).values()

print(f"Training Set Size: {len(train_dataset)}")
print(f"Validation Set Size: {len(validation_dataset)}")
print(f"Test Set Size: {len(test_dataset)}")

# Define metrics
def compute_metrics(pred):
    """
    Purpose: Calculate evaluation metrics for model predictions.
    Input: pred (PredictionOutput): Predictions and true labels.
    Output: metrics (dict): Dictionary of calculated metrics.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate(test_dataset)
print("\nFinal Evaluation Results:", eval_results)

# Process predictions
processed_data = []
for item in test_dataset:
    inputs = tokenizer(item["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = unique_labels[torch.argmax(outputs.logits).item()]
    processed_data.append({"prediction": predicted_label, "target": unique_labels[item["label"]]})

# Tag-specific metrics
tag_metrics = defaultdict(list)
for pred in processed_data:
    tag_metrics[pred["target"].capitalize()].append(1 if pred["prediction"] == pred["target"] else 0)

print("\nTag-Specific Metrics:")
for tag, scores in tag_metrics.items():
    print(f"{tag} Accuracy: {np.mean(scores):.4f}")
