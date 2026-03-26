import torch
import torch.nn as nn

from config import *
from data import load_data
from model import Femto_Chatbot


def generate_new_sentences():

    X, _, dic = load_data(FILENAME, CONTEXT_SIZE)
    voc_size = len(dic)
    decoding = {i: word for word, i in dic.items()}

    model = Femto_Chatbot(voc_size, EMBEDDING_DIM)
    model.load_state_dict(torch.load("femto_chatbot.pt"))

    sentence = X[0]

    actual_sentence = ""
    for i in range(0, 15):
        logits = model(torch.tensor(sentence))[-1]
        prob = nn.Softmax(dim=-1)(logits)
        sample = torch.multinomial(prob, 1).item()
        actual_sentence += " " + decoding[sample]
        sentence = sentence[1:] + [sample]
    return actual_sentence


if __name__ == "__main__":
    for i in range(0, 10):
        print(generate_new_sentences())
