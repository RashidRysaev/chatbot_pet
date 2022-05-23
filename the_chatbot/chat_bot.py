import json
import random

import torch

from model import NeuralNetModel
from nltk_utils import bag_of_words, tokenize

with open("intents.json", "r") as f:
    intents = json.load(f)

file = "data.pth"
data = torch.load(file)

input_size = data.get("input_size")
output_size = data.get("output_size")
hidden_size = data.get("hidden_size")
all_words = data.get("all_words")
model_state = data.get("model_state")
tags = data.get("tags")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


# chat:
bot_name = "Mate_Bot"

print("Chat is on! Type 'Q' to quit the chatbot")
while True:
    user_input = input("You: ")

    if user_input == "Q":
        print("Chat is off!")
        break

    sentence = tokenize(user_input)
    bag = bag_of_words(sentence, all_words)
    res = bag.reshape(1, bag.shape[0])
    res = torch.from_numpy(res)

    output = model(res)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    if probability.item() > 0.75:
        for intent in intents.get("intents"):
            if tag == intent.get("tag"):
                print(f'{bot_name}: {random.choice(intent.get("responses"))}')  # bot response
    else:
        print(f"{bot_name}: Sorry, I don`t get you. Could you reword your question or ask something else?")
