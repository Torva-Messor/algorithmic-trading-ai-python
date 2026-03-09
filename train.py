import json
import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
from models.gemma_transformer_classifier import SimpleGemmaTransformerClassifier
from sklearn.metrics import f1_score 

## Parameters
learning_rate = 0.005
batch = 1
epochs = 20

## Define our labels
sell = [1., 0., 0.]
hold = [0., 1., 0.]
buy  = [0., 0., 1.]

## Model
model = SimpleGemmaTransformerClassifier()
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

## Read JSON data
with open('BTC-USD_news_with_price.json', 'r') as f:
    training = json.load(f)

## Generate our features and labels
features = []
labels = []

for item in training:
    ## Format the metrics cleanly
    vix_val = f"{item['global_vix']:.2f}" if isinstance(item['global_vix'], (int, float)) else item['global_vix']
    dxy_val = f"{item['macro_dxy']:.2f}" if isinstance(item['macro_dxy'], (int, float)) else item['macro_dxy']
    tnx_val = f"{item['macro_tnx']:.3f}%" if isinstance(item['macro_tnx'], (int, float)) else item['macro_tnx']
    vol_val = f"{item['local_volatility']:.2f}" if isinstance(item['local_volatility'], (int, float)) else "0.00"

    ## Inject Macro, Micro, and News context into the LLM's prompt
    features.append(
        "\n".join([
            f"Macro Environment: US Dollar Index (DXY) is {dxy_val}. 10-Year Treasury Yield is {tnx_val}.",
            f"Market Context: Global Volatility (VIX) is {vix_val}. Local BTC 1-hour volatility is {vol_val}.",
            f"Current BTC Price: {item['price']}",
            f"Headline: {item['title']}",
            f"Summary: {item['summary']}"
        ])
    )

    ## Generate labels based on percentage change
    ## Actions: Buy / Sell / Hold
    if item['percentage'] < -0.01:
        labels.append(sell)
    elif item['percentage'] > 0.01:
        labels.append(buy)
    else:
        labels.append(hold)

## Separate Train and Test sets
split_index = int(0.8 * len(features))
train_features = features[:split_index]
train_labels = labels[:split_index]
test_features = features[split_index:]
test_labels = labels[split_index:]

## Training Loop
item_losses = []
model.train()
for epoch in range(epochs):
    stochastic = np.random.permutation(len(train_features))
    inputs = np.array(train_features)[stochastic]
    targets = np.array(train_labels)[stochastic]

    for i in range(len(inputs) // batch):
        input_batch  = inputs[i * batch : i * batch + batch]
        target_batch = torch.from_numpy(targets[i * batch : i * batch + batch])

        optimizer.zero_grad()
        logits = model(input_batch)

        loss = criterion(
            logits,
            target_batch.float().to(torch.device('mps'))
        )
        loss.backward()
        optimizer.step()
        
        probs = logits.softmax(dim=-1).detach().cpu()
        item_losses.append(loss.item())
        cost = sum(item_losses[-250:]) / len(item_losses[-250:])
        print(f"Epoch {epoch + 1}: loss={cost:.4f} probs={probs}")

## Evaluate Model Accuracy
correct = 0
total = 0
all_predictions = []
all_actuals = []

model.eval()
with torch.no_grad():
    for i in range(len(test_features)):
        input_text  = [test_features[i]]
        target = torch.tensor(test_labels[i])

        logits = model(input_text)
        probs = logits.softmax(dim=-1).cpu()
        predicted = torch.argmax(probs, dim=-1)
        actual = torch.argmax(target.float().to(torch.device('mps')))

        all_predictions.append(predicted.item())
        all_actuals.append(actual.item())

        if predicted.item() == actual.item():
            correct += 1
        total += 1

## Calculate Accuracy and F1 Score
accuracy = correct / total
f1 = f1_score(all_actuals, all_predictions, average='weighted')

print(f"\nEvaluation Results:")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test F1 Score (Weighted): {f1:.4f}")
