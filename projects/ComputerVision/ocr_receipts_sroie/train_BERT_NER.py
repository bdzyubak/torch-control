import torch
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load your data and split it into train and validation sets
# Replace `texts` and `labels` with your data
texts = [...]  # List of text data
labels = [...]  # List of corresponding labels (NER tags)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Tokenize the texts and labels
train_encodings = tokenizer(train_texts, is_split_into_words=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, padding=True, truncation=True)

# Convert labels to ids
label2id = {label: i for i, label in enumerate(set(label for label_list in train_labels for label in label_list))}
id2label = {i: label for label, i in label2id.items()}

train_labels_ids = [[label2id[label] for label in label_list] for label_list in train_labels]
val_labels_ids = [[label2id[label] for label in label_list] for label_list in val_labels]

# Load pre-trained BERT model for token classification
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label2id))

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy='steps',
    save_total_limit=2,
)

# Define Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=list(zip(train_encodings['input_ids'], train_labels_ids)),
    eval_dataset=list(zip(val_encodings['input_ids'], val_labels_ids)),
    tokenizer=tokenizer,
    compute_metrics=lambda p: {'accuracy': torch.tensor(0)}  # We won't use accuracy for NER
)

# Fine-tune the model
trainer.train()