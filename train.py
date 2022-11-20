import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pyts.datasets import fetch_ucr_dataset as fetch_dataset
from pathlib import Path
import argparse
from triplet_dataset import TripletDataset
from resnet import ResNet
import torch
import torch.nn as nn
import os

def load_UEA_dataset(dataset, train_ratio=0.9,random_state=0, path = None):
    """
    Loads the UEA dataset given in input in np arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    # Initialization needed to load a file with Weka wrappers_test
    if path is None:
        train, test, train_labels, test_labels = fetch_dataset(dataset, return_X_y=True)
    else:
        train, test, train_labels, test_labels = fetch_dataset(dataset, return_X_y=True, data_home=str(path) + '/')
        
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

    if train_ratio < 1:
        X_train_ori, y_train_ori = train, train_labels
        sss = StratifiedShuffleSplit(n_splits=10, test_size=1 - train_ratio, random_state=random_state)
        sss.get_n_splits(X_train_ori, y_train_ori)

        for train_index, test_index in sss.split(X_train_ori, y_train_ori):
            train = X_train_ori[train_index,:]
            train_labels = y_train_ori[train_index]
    
        print(f'train shape: {np.shape(train)}')
    print(f'dataset load succeed for random state {random_state}!!!')
    return train, train_labels, test, test_labels

def validate(model, test_loader, criterion, device):
    print('validating...')
    model.eval()
    test_loss = 0
    correct = 0
    disance_metric = nn.PairwiseDistance(p=2)
    with torch.no_grad():
        for data, _ in test_loader:
            anchor, positive, negative = data
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor), model(positive), model(negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            positive_dist = disance_metric(anchor_output, positive_output)
            negative_dist = disance_metric(anchor_output, negative_output)
            batch_correct = torch.sum(positive_dist < negative_dist).item()
            correct += batch_correct
            test_loss += loss.item()


    test_loss /= len(test_loader.dataset)
    print(f'Val set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
    return test_loss, correct / len(test_loader.dataset)

def test(model, test_loader, train_loader, device):
    model.eval()
    # get support embeddings
    support_embeddings = []
    support_labels = []
    for data, label in train_loader:
        data = data[0].to(device)
        support_embeddings.append(model(data))
        support_labels.append(label)
    support_embeddings = torch.cat(support_embeddings, dim=0)
    # get mean embedding for each class
    support_labels = torch.cat(support_labels, dim=0)
    support_embeddings = support_embeddings.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()
    mean_embeddings = []
    temp_labels = []
    for label in np.unique(support_labels):
        mean_embeddings.append(np.mean(support_embeddings[support_labels == label], axis=0))
        temp_labels.append(label)
    support_embeddings = torch.from_numpy(np.array(mean_embeddings)).to(device)
    support_labels = torch.from_numpy(np.array(temp_labels)).to(device)
    # get query embeddings
    query_embeddings = []
    query_labels = []
    for data, label in test_loader:
        data = data[0].to(device)
        query_embeddings.append(model(data))
        query_labels.append(label)
    query_embeddings = torch.cat(query_embeddings, dim=0)
    query_labels = torch.cat(query_labels, dim=0)
    # calculate accuracy
    disance_metric = nn.PairwiseDistance(p=2)
    correct = 0
    for i in range(len(query_embeddings)):
        dist = disance_metric(query_embeddings[i], support_embeddings)
        min_index = torch.argmin(dist).item()
        if support_labels[min_index] == query_labels[i]:
            correct += 1
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
    return correct / len(test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ResNet Training') 
    parser.add_argument('--dataset', default='names.txt', type=str, help='dataset name')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='train ratio')
    parser.add_argument('--random_state', default=0, type=int, help='random state')
    parser.add_argument('--path', default='./data', type=str, help='path to dataset')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=2000, type=int, help='epochs')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--test_interval', default=10, type=int, help='test interval')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    path = Path(args.path)
    save_path = Path('./results')

    dataset = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    epochs = args.epochs
    lr = args.lr
    test_interval = args.test_interval
    batch_size = args.batch_size

    if '.txt' in dataset:
        with open(dataset,'r') as f:
            datasets = f.read().splitlines()
    else:
        datasets = [dataset]

    for dataset in datasets:  

        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True)
        csv_file = Path(str(save_path), str(dataset) + '.csv')

        if csv_file.exists():
            df = pd.read_csv(csv_file)

        else:
            df = pd.DataFrame(columns=['ratio', 'random_state', 'accuracy'])
            df.to_csv(csv_file, index=False)

        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]:
            for random_state in range(3):

                print(f'dataset:{dataset} ratio: {ratio}, random_state: {random_state}')

                if ratio == 1 & random_state != 0:
                    continue
                try:
                    if not df[(df['ratio'] == ratio) & (df['random_state'] == random_state)].empty:

                        print('Already done')

                        continue
                            
                    train_data, train_labels, test_data, test_labels = load_UEA_dataset(dataset, train_ratio=ratio,random_state=random_state, path = path)
                    train_data = np.expand_dims(train_data, axis=1)
                    n_classes = len(np.unique(train_labels))

                    print(f'number of classes: {n_classes}')

                    # split train data into train and validation
                    X_train_ori, y_train_ori = train_data, train_labels

                    print(f'train labels: {y_train_ori}')

                    test_size = 0.2 if len(X_train_ori) // n_classes >= 10 else n_classes * 2

                    print(f'test_size: {test_size}')

                    sss = StratifiedShuffleSplit(n_splits=10, test_size=test_size, random_state=0)
                    sss.get_n_splits(X_train_ori, y_train_ori)
                    for train_index, test_index in sss.split(X_train_ori, y_train_ori):
                        train_data, val_data = X_train_ori[train_index,:], X_train_ori[test_index,:]
                        train_labels, val_labels = y_train_ori[train_index], y_train_ori[test_index]

                    
                    print(f'train shape: {np.shape(train_data)}')
                    print(f'train labels: {train_labels}')
                    if np.any(np.bincount(train_labels) < 2):
                        print('train labels are not continuous')
                        continue
                    print(f'val shape: {np.shape(val_data)}')
                    print(f'val labels: {val_labels}')

                    # if (np.shape(train_data)[0] // n_classes) < 2 or (np.shape(val_data)[0] // n_classes) < 2:
                    if (np.shape(train_data)[0] // n_classes) < 2:
                        print('Not enough data')
                        break
                    
                    test_data = np.expand_dims(test_data, axis=1)
                    train_dataset = TripletDataset(train_data, train_labels)
                    val_dataset = TripletDataset(val_data, val_labels, train=False)
                    test_dataset = TripletDataset(test_data, test_labels, train=False)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                    
                    model = ResNet()
                    # move to GPU
                    model.to(device)
                    
                    ## train model
                    loss = nn.TripletMarginLoss(margin=1.0, p=2)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    mod = 'train'
                    best_model = None
                    best_loss = 100000
                    losses = []

                    print('Start training')
                    print('Initialize test')

                    # test(model, test_loader, train_loader, device)
                    for epoch in range(epochs):
                        model.train()

                        for i, (data,_) in enumerate(train_loader):

                            anchor, positive, negative= data
                            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                            optimizer.zero_grad()
                            anchor = anchor.float()
                            positive = positive.float()
                            negative = negative.float()

                            anchor = model(anchor)
                            positive = model(positive)
                            negative = model(negative)

                            triplet_loss = loss(anchor, positive, negative)
                            triplet_loss.backward()
                            optimizer.step()

                        print(f'epoch {epoch} loss {triplet_loss}')

                        if epoch % test_interval == 0 or epoch == epochs - 1:
                            test_loss, test_acc = validate(model, val_loader, loss, device)
                            if test_loss < best_loss:
                                best_loss = test_loss
                                best_model = model
                                # torch.save(best_model.state_dict(), 'best_model.pt')

                                print('best model changed')

                            losses.append(test_loss)

                            if len(losses) > 3 and losses[-1] >= losses[-2] >= losses[-3]:
                                print('Early stopping')
                                break

                    print('Start testing')

                    accuracy = test(best_model, test_loader, train_loader, device)
                    with open(csv_file, 'a') as f:
                        f.write(f'{ratio},{random_state},{accuracy}\n')


                except Exception as e:
                    print(e)
                    print('ratio {} random_state {} failed'.format(ratio, random_state))
                    print(e)
                    continue