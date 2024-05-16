import torch
import pickle as pkl
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader

from dataset import CorpusData
from model import FastText


def train(model, epoch, train_loader, optimizer, criterion, device):
    model.train()
    avg_loss = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.append(loss.item())
        if i%10 == 0:
            print('epoch:{}, iter:{}/{}, train loss:{:.4f}'.format(epoch, i, len(train_loader), sum(avg_loss) / len(avg_loss)))

    print('epoch:{}, train loss:{:.4f}'.format(epoch, sum(avg_loss) / len(avg_loss)))

def eval(model, val_loader, device):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    return metrics.accuracy_score(labels_all, predict_all)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vocab_path = 'data/vocab.pkl'
    train_path = 'data/train.txt'
    test_path = 'data/test.txt'
    class_path = 'data/class.txt'

    sentence_len = 32
    embed_size = 300
    n_gram_vocab = 250499 # hash

    vocab = pkl.load(open(vocab_path, 'rb'))  # 4762
    n_vocab = len(vocab)
    class_list = [x.strip() for x in open(class_path, encoding='utf-8').readlines()]
    num_classes = len(class_list)

    learning_rate = 1e-3
    batch_size = 128
    num_epochs = 20

    # def data
    train_dataset = CorpusData(train_path, vocab, sentence_len, n_gram_vocab)
    val_dataset = CorpusData(test_path, vocab, sentence_len, n_gram_vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn)

    # def model
    model = FastText(n_vocab, n_gram_vocab, embed_size, num_classes)
    model = model.to(device)

    # def optim
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # def loss func
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # train
    best_acc = 0.
    for epoch in range(num_epochs):
        train(model, epoch, train_loader, optimizer, criterion, device)
        with torch.no_grad():
            acc = eval(model, val_loader, device)
            best_acc = acc if acc > best_acc else best_acc
            print('epoch:{}, test acc:{:.4f}, best acc:{:.4f}'.format(epoch, acc, best_acc))

    torch.save(model.state_dict(), 'final.pth')