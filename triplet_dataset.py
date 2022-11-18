
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, data_x = None, labels = None, train = True):
        self.train = train
        if self.train:
            self.labels = torch.LongTensor(labels)
            self.data_x = torch.FloatTensor(data_x)
            self.labels_set = set(np.unique(self.labels))
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.labels = torch.LongTensor(labels)
            self.data_x = torch.FloatTensor(data_x)
            # generate fixed triplets for testing
            self.labels_set = set(np.unique(self.labels))
            self.label_to_indices = {label: np.where(self.labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.data_x))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            anchor, label1 = self.data_x[index], self.labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            pos = self.data_x[positive_index]
            neg = self.data_x[negative_index]
        else:
            anchor = self.data_x[self.test_triplets[index][0]]
            label1 = self.labels[self.test_triplets[index][0]].item()
            pos = self.data_x[self.test_triplets[index][1]]
            neg = self.data_x[self.test_triplets[index][2]]

        # anchor = torch.FloatTensor(anchor)
        # pos = torch.FloatTensor(pos)
        # neg = torch.FloatTensor(neg)
        return (anchor, pos, neg), label1

    def __len__(self):
        return len(self.labels) if self.train else len(self.test_triplets)
