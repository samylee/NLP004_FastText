from tqdm import tqdm
from torch.utils.data import Dataset
import torch


class CorpusData(Dataset):
    def __init__(self, data_path, vocab, sentence_len, n_gram_vocab):
        super(CorpusData, self).__init__()
        tokenizer = lambda x: [y for y in x]  # char-level
        UNK, PAD = '<UNK>', '<PAD>'

        self.contents = []
        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)

                if len(token) < sentence_len:
                    token.extend([PAD] * (sentence_len - len(token)))
                else:
                    token = token[:sentence_len]

                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # fasttext ngram
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(sentence_len):
                    bigram.append(self.biGramHash(words_line, i, n_gram_vocab))
                    trigram.append(self.triGramHash(words_line, i, n_gram_vocab))
                # -----------------
                self.contents.append((words_line, bigram, trigram, int(label)))

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        data = self.contents[index]

        x = torch.LongTensor(data[0])
        bigram = torch.LongTensor(data[1])
        trigram = torch.LongTensor(data[2])

        image = torch.stack([x, bigram, trigram])
        label = torch.LongTensor([data[3]])

        return image, label

    def biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets # 250499

    def collate_fn(self, batch):
        images, targets = list(zip(*batch))

        x = torch.stack([image[0] for image in images])
        bigram = torch.stack([image[1] for image in images])
        trigram = torch.stack([image[2] for image in images])

        images = torch.stack([x, bigram, trigram])
        targets = torch.stack(targets).squeeze()

        return images, targets