# Sentiment Analysis using LSTM

This project focuses on analyzing the sentiment of text data using deep learning models like Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to capture the temporal dependencies in sequential data such as text.


## Code

To run this project, you can either run in jupyter notebook or vs code. Firstly install the modules and import them.

```bash
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
```
We use the IMDB dataset to train our model, which can be found in kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews. We need to preprocess the dataset before training the model.
```bash
data = pd.read_csv('IMDB Dataset.csv')
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
data['review'] = data['review'].apply(lambda x: x.lower())
data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
data['review'] = data['review'].apply(str)
```
Then we use sentence transformer to encode the sentences, as using word embeddings can make the process of training inefficient.
```bash
sentence_model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
reviews = data['review'].tolist()
sentiments = data['sentiment'].tolist()
numerical_reviews = sentence_model.encode(reviews)
x_train, x_test, y_train, y_test = train_test_split(numerical_reviews, sentiments, test_size=0.2, random_state=42)

```
We create training and testing dataset, and use batch_size of 32 as it makes training process more easier.
```bash
x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
```
Here's the blue print of our LSTM model
```bash
class Sentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super(Sentiment, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        output, (hidden, cell) = self.lstm(inputs)
        hidden = self.dropout(torch.cat((hidden[-1, :, :].unsqueeze(0),), dim=1))
        return self.fc(hidden.squeeze(0))
```
Here we specify the hyperparameters and input dimensions etc.
```bash
input_dim, hidden_dim, output_dim, n_layers, dropout = 768, 256, 2, 1, 0.5
model = Sentiment(input_dim, hidden_dim, output_dim, n_layers, dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```
This code trains the model, i have used H100 SXM GPU to train, and it took around 2 mins to train. I suggest you to use cloud GPU.
```bash
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{10}, Loss: {loss.item():.4f}')
```
This code calculates the accuracy of the model using the test dataset.
```bash
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
```
We check the model performance by passing random input, which is also called inference.
```bash
#inference
input_text = "I'm very angry at you"
encoded_input = torch.tensor(sentence_model.encode(input_text)).unsqueeze(0)  # Add batch dimension
encoded_input = encoded_input.to(device)  # Move input to device if necessary

with torch.no_grad():  
    model.eval()  # Set the model to evaluation mode
    output = model(encoded_input)  # Pass input through the model

# Convert output to probabilities
probabilities = torch.softmax(output, dim=1)

# Get the predicted sentiment (assuming binary classification)
predicted_sentiment = torch.argmax(probabilities, dim=1).item()

# Print the predicted sentiment
if predicted_sentiment == 1:
    print("Positive sentiment")
else:
    print("Negative sentiment")
```
## License

[MIT](https://choosealicense.com/licenses/mit/)


