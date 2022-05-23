import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chat_class import ChatDataset
from model import NeuralNetModel
from training import all_words, tags, x_train, y_train

# hyperparameters:
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])  # length of each bag of words created above; has the same length of `all_words` array
learning_rate = 0.001
num_epochs = 1000


chat_dataset = ChatDataset()
train_loader = DataLoader(dataset=chat_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetModel(input_size, hidden_size, output_size).to(device)

# loss and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop:
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward:
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # step checker:
    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch + 1}/{num_epochs} \n loss: {loss.item():.4f}")
print(f"final loss: {loss.item():.4f}")


# save the data:
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}

file = "data.pth"
torch.save(data, file)
print(f"Training has been completed! Data saved to {file}")
