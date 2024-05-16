import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, n_vocab, n_gram_vocab, embed_size, num_classes):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_size, padding_idx=n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed_size)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed_size)
        self.fc1 = nn.Linear(embed_size * 3, 256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[1])
        out_trigram = self.embedding_ngram3(x[2])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)

        out = self.relu1(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out