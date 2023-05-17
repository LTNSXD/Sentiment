import torch
import gensim
import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, filepath):
        # 导入词向量模型
        Word2VecModel = "Dataset/wiki_word2vec_50.bin"
        self.PreModel = gensim.models.keyedvectors.load_word2vec_format(Word2VecModel, binary=True)
        self.key2index = self.PreModel.key_to_index
        # 句长固定为50
        SentenceLen = 50
        try:
            with open(filepath, encoding="utf-8") as file:
                lines = file.readlines()
                self.label = [int(line.split('\t', 1)[0]) for line in lines]
                content = [line.split('\t', 1)[1][:-1] for line in lines]
                sentences = [sentence.split(' ') for sentence in content]
                self.embedding_sentences = []
                for sentence in sentences:
                    m_sentence = []
                    for word in sentence:
                        try: 
                            m_sentence.append(self.key2index[word])
                        except:
                            pass
                            # m_sentence.append(np.random.randint(420000))
                    if len(m_sentence) >= SentenceLen:
                        m_sentence = m_sentence[:SentenceLen]
                    else: # padding
                        m_sentence += [110] * (SentenceLen - len(m_sentence))
                    m_sentence = torch.tensor(m_sentence)
                    self.embedding_sentences.append(m_sentence)
                assert len(self.label) == len(self.embedding_sentences)
                self.len = len(self.label)

                self.label = torch.tensor(self.label)
                self.embedding_sentences = torch.stack(self.embedding_sentences, dim=0)
 
        except Exception as e:
            print(e)

    def __getitem__(self, index):
        assert 0 <= index <= self.len, "Illegal index"
        return self.embedding_sentences[index], self.label[index]

    def __len__(self):
        return self.len
