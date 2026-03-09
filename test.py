import json
import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
from sklearn.metrics import f1_score 

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Testing on device: {device}")

## Define our labels "tokens"
sell = [1., 0., 0.]
hold = [0., 1., 0.]
buy  = [0., 0., 1.]

model = SimpleGemmaTransformerClassifier()


model.load_state_dict(torch.load('gemma_transformer_classifier.pth', map_location=device, weights_only=True))
model = model.to(device) # Move model to detected device

## read BTC-USD_news_with_price.json
with open('BTC-USD_news_with_price.json', 'r') as f:
    training = json.load(f)

## Generate our features and labels
features = []
labels = []

for item in training:
    features.append(
        "\n".join([f"Price: {item['price']}",
        f"Headline: {item['title']}",
        f"Summary: {item['summary']}"])
    )

    ## Generate labels based on percentage change
    ## Actions Buy / Sell / Hold
    if item['percentage'] < -0.01:
        labels.append(sell)
    elif item['percentage'] > 0.01:
        labels.append(buy)
    else:
        labels.append(hold)

## Separate Train and Test sets
split_index = int(0.8 * len(features))
test_features = features[split_index:]
test_labels = labels[split_index:]

## Evaluate Model Accuracy
correct = 0
total = 0
all_predictions = []
all_actuals = []

model.eval()
with torch.no_grad():
    for i in range(len(test_features)):
        input  = [test_features[i]]
        
        target = torch.tensor(test_labels[i]).float().to(device)

        logits = model(input)
        probs = logits.softmax(dim=-1).cpu()
        predicted = torch.argmax(probs, dim=-1)
        actual = torch.argmax(target)

        all_predictions.append(predicted.item())
        all_actuals.append(actual.item())

        if predicted.item() == actual.item():
            correct += 1
        total += 1

# Calculate Accuracy
print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")

# Calculate F1 Score
f1 = f1_score(all_actuals, all_predictions, average='weighted')
print(f"F1 Score (weighted): {f1:.4f}")
print("Done")

print("Saving model...")
torch.save(model.state_dict(), 'gemma_transformer_classifier.pth')
