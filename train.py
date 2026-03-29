import torch
import torch.nn as nn
from config import *
from data import load_data
from model import Femto_Chatbot


def train(model, X, Y, num_epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr)
    print("Converting to tensors...")
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            batch_Y = Y[i : i + batch_size]

            logits = model(batch_X)
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, voc_size),  # (batch*seq, voc)
                batch_Y.view(-1),  # (batch*seq,)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


if __name__ == "__main__":
    filename = FILENAME

    print("Loading data...")
    X, Y, dic = load_data(filename, CONTEXT_SIZE)
    print(f"Data loaded: {len(X)} examples, vocab size: {len(dic)}")

    voc_size = len(dic)
    model = Femto_Chatbot(voc_size, CONTEXT_SIZE, EMBEDDING_DIM)

    print("Model created, starting training...")
    train(model, X, Y, NUM_EPOCHS, BATCH_SIZE, LR)

    torch.save(model.state_dict(), "femto_chatbot.pt")
