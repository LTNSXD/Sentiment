import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import TextDataset
from config import *
from sklearn.metrics import f1_score

def test(dataloader, model, accuracies) -> float:
    model.eval()
    segment_true = []
    segment_pred = []
    with torch.no_grad():
        for sentence, label in dataloader:
            prediction = model(sentence)
            result = torch.max(prediction, dim=1)[1]
            accuracies.append(torch.eq(result, label).float().mean())
            segment_true.append(label)
            segment_pred.append(result)
        segment_true = torch.cat(segment_true, dim=0)
        segment_pred = torch.cat(segment_pred, dim=0)
        return f1_score(segment_true, segment_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training models')
    parser.add_argument('--m', type=str, default="CNN", help='type of current model')
    args = parser.parse_args()

    m = args.m

    if m == "CNN":
        model_path = "models/CNN.pt"
    elif m == "RNN":
        model_path = "models/RNN.pt"
    elif m == "DNN":
        model_path = "models/DNN.pt"
    else:
        print("Usage: python code/test.py --m [model], where model could be CNN, RNN, DNN")
        sys.exit(0)

    accuracies = []
    test_path = "Dataset/test.txt"
    dataset = TextDataset(test_path)
    textLoader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model = torch.load(model_path)
    score = test(textLoader, model, accuracies)

    accuracy = np.array(accuracies).mean()

    print("test_acc: {}, f1_score: {}".format(accuracy, score))
    
