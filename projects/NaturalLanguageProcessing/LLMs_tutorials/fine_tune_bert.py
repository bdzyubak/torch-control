from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn as nn

# Tutorial Courtesy https://www.analyticsvidhya.com/blog/2023/08/fine-tuning-large-language-models/

# Load the pre-trained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the pre-trained model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

texts = ["I loved the movie. It was great!",
         "The food was terrible.",
         "The weather is okay."]
sentiments = ["positive", "negative", "neutral"]

# Tokenize the text samples
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Extract the input IDs and attention masks
input_ids = encoded_texts['input_ids']
attention_mask = encoded_texts['attention_mask']

# Convert the sentiment labels to numerical form
sentiment_labels = [sentiments.index(sentiment) for sentiment in sentiments]

# Add a custom classification head on top of the pre-trained model
num_classes = len(set(sentiment_labels))
classification_head = nn.Linear(model.config.hidden_size, num_classes)

# Replace the pre-trained model's classification head with our custom head
model.classifier = classification_head
