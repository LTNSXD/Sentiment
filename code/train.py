import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from config import *
from model import CNN, RNN_LSTM, DNN
from dataset import TextDataset
from test import test

def train(dataloader, model):
    """
    封装的单次训练函数
    """
    model.train()
    for index, data in enumerate(dataloader):
        sentence, label = data
        # Forward
        pred_label = model(sentence)
        loss = criterion(pred_label, label)
        # loss = F.nll_loss(pred_label, label)
        result = torch.max(pred_label, dim=1)[1]
        accuracy = torch.eq(result, label).float().mean()
        train_accuracies.append(accuracy)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Upgrade
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training models')
    parser.add_argument('--m', type=str, default="CNN", help='type of current model')
    args = parser.parse_args()

    m = args.m

    if m == "CNN":
        model = CNN()
        save_path = "models/CNN.pt"
        EPOCH = 10
    elif m == "RNN":
        model = RNN_LSTM()
        save_path = "models/RNN.pt"
        EPOCH = 5
    elif m == "DNN":
        model = DNN()
        save_path = "models/DNN.pt"
        EPOCH = 5
    else:
        print("Usage: python code/train.py --m [model], where model could be CNN, RNN, DNN")
        sys.exit(0)

    train_path = "Dataset/train.txt"
    val_path = "Dataset/validation.txt"

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), LEARNING_RATE)

    train_dataset = TextDataset(train_path)
    val_dataset = TextDataset(val_path)
    trainLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 训练
    for epoch in range(EPOCH):
        train_accuracies = []
        train(trainLoader, model)
        train_average = np.array(train_accuracies).mean()
        val_accuracies = []
        test(valLoader, model, val_accuracies)
        val_average = np.array(val_accuracies).mean()
        print("Epoch{epoch}: train_acc: {train_average}, val_acc: {val_average}".format(epoch=epoch+1, train_average=train_average, val_average=val_average))

    # Save the model
    torch.save(model, save_path)

