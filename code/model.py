import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 导入词向量
        Word2VecModel = "Dataset/wiki_word2vec_50.bin"
        self.PreModel = gensim.models.keyedvectors.load_word2vec_format(Word2VecModel, binary=True)
        self.vectors = self.PreModel.vectors

        # 初始化
        # 输入通道数（情感分类问题下为一个通道）
        self.in_channels = IN_CHANNELS
        # 输出通道数
        self.out_channels = OUT_CHANNELS
        # 词向量维度
        self.embedding_dim = EMBEDDING_DIM
        # 卷积核尺寸
        self.filter_sizes = FILTER_SIZES
        # 分类数（二分类问题下为2）
        self.categories = CATEGORIES
        # 词向量个数
        self.vocabulary_size = len(self.vectors)
        # Embedding层
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in self.vectors])
        self.embedding.weight.data.copy_(tensor_embedding)

        # 卷积层
        self.convs = [nn.Conv2d(self.in_channels, self.out_channels, (fs, self.embedding_dim)) for fs in self.filter_sizes]
        # 全连阶层
        self.fc = nn.Linear(len(self.filter_sizes) * self.out_channels, self.categories)
        # dropouts
        self.dropout = nn.Dropout(DROPOUT)

    def conv_and_pool(self, x, conv):
        x = x.unsqueeze(1)
        x = conv(x).squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        pooled = [self.conv_and_pool(x, conv) for conv in self.convs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        output = self.fc(cat)
        return output


class RNN_LSTM(nn.Module):
    def __init__(self):
        super(RNN_LSTM, self).__init__()

        # 导入词向量
        Word2VecModel = "Dataset/wiki_word2vec_50.bin"
        self.PreModel = gensim.models.keyedvectors.load_word2vec_format(Word2VecModel, binary=True)
        self.vectors = self.PreModel.vectors

        # 初始化
        # 词向量维度
        self.embedding_dim = EMBEDDING_DIM
        # 分类数（二分类问题下为2）
        self.categories = CATEGORIES
        # 词向量个数
        self.vocabulary_size = len(self.vectors)
        # Embedding层
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in self.vectors])
        self.embedding.weight.data.copy_(tensor_embedding)

        self.hidden_size = HIDDEN_SIZE
        self.hidden_layers = HIDDEN_LAYERS

        # LSTM
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.hidden_layers, bidirectional=True)
        self.decoder = nn.Linear(2 * self.hidden_size, 64)
        # 全联接层
        self.fc = nn.Linear(64, self.categories)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(self.embedding(x).permute(1, 0, 2))
        h_n = h_n.view(self.hidden_layers, 2, -1, self.hidden_size)
        cat = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        return self.fc(self.decoder(cat))


class DNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 导入词向量
        Word2VecModel = "Dataset/wiki_word2vec_50.bin"
        self.PreModel = gensim.models.keyedvectors.load_word2vec_format(Word2VecModel, binary=True)
        self.vectors = self.PreModel.vectors

        # 初始化
        # 词向量维度
        self.embedding_dim = EMBEDDING_DIM
        # 分类数（二分类问题下为2）
        self.categories = CATEGORIES
        # 词向量个数
        self.vocabulary_size = len(self.vectors)
        # Embedding层
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        tensor_embedding = torch.stack([torch.from_numpy(array).float() for array in self.vectors])
        self.embedding.weight.data.copy_(tensor_embedding)

        self.linear1 = nn.Linear(50 * 50, 30)
        self.linear2 = nn.Linear(30, 10)
        self.linear3 = nn.Linear(10, self.categories)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, 50 * 50)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return F.sigmoid(x)
