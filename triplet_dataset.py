
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, train_data = None, train_labels = None, test_data = None, test_labels = None,  train = True):
        self.train = train
        if self.train:
            self.train_labels = torch.LongTensor(train_labels)
            self.train_data = torch.FloatTensor(train_data)
            self.labels_set = set(np.unique(self.train_labels))
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = torch.LongTensor(test_labels)
            self.test_data = torch.FloatTensor(test_data)
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            anchor, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            pos = self.train_data[positive_index]
            neg = self.train_data[negative_index]
        else:
            anchor = self.test_data[self.test_triplets[index][0]]
            label1 = self.test_labels[self.test_triplets[index][0]].item()
            pos = self.test_data[self.test_triplets[index][1]]
            neg = self.test_data[self.test_triplets[index][2]]

        # anchor = torch.FloatTensor(anchor)
        # pos = torch.FloatTensor(pos)
        # neg = torch.FloatTensor(neg)
        return (anchor, pos, neg), label1

    def __len__(self):
        return len(self.train_labels) if self.train else len(self.test_triplets)
