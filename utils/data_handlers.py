import copy

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
import random


def shuffle_online_data(data_folder, n_shuffles, classes):
    obj_labels = []
    online_arr = np.empty((0, 190))
    object_array = np.empty((0, 190))
    for obj_name in classes:
        object_array = np.empty((0, 190))
        # load the deltas file of the object required
        data_file = f"{data_folder}val_test_grasps/{obj_name}_deltas"
        with open(data_file, "rb") as fp:
            input_data = pickle.load(fp)
        # online_arr = input_data
        obj_labels += [obj_name] * n_shuffles * 2

        input_arr = np.reshape(np.concatenate(input_data), (-1, 190))
        for shuffle in range(n_shuffles):
            a = input_arr
            noise = np.random.normal(-0.5, 0.5, a.shape)
            a = a + noise * a / 5
            a[a < 2] = 0
            np.random.shuffle(a.reshape([-1, 19]))  # 'this is looking to randomise the order of the grasps'
            online_arr = np.concatenate((online_arr, a), axis=0)
        object_array = np.concatenate((object_array, online_arr), axis=0)
    np.save(f'{data_folder}shuffled_online_data.npy', object_array)
    np.save(f'{data_folder}shuffled_online_labels.npy', obj_labels)


class ObjectGraspsDataset(Dataset):
    """Object Grasps Dataset"""

    def __init__(self, data_array_file, labels_file, n_grasps, data_max=None, data_min=None, transform=None,
                 train: bool = True, pre_sort=False, random_pad=True):
        """
        Args:
            array_file (string): Path to the saved ndarray file with grasp raw data.
            labels_file (string): Path to the saved ndarray file with labels.
            n_grasps (int): number of grasps to be trained on for data reshuffle
        """

        input_data = np.load(data_array_file)
        input_targets = np.load(labels_file)
        shuffled_data = np.empty((0, 1, 10, 19))
        shuffled_labels = []
        labels = list(set(input_targets))
        # reshape data to align sensors but maintain label for each grasp
        for label in labels:
            d_label = input_data[label == input_targets]
            # if n_grasps != 7:
            #    d_shuffled = d_label.reshape((-1, 1, n_grasps, 19))
            # else:
            dim_len = int(np.floor(len(d_label) / n_grasps))
            d_shuffled = d_label[0:dim_len * n_grasps, :].reshape((-1, 1, n_grasps, 19))
            if random_pad:
                padded_rows = np.random.randint(1, n_grasps)
                d_shuffled[:, :, padded_rows:, :] = 0
                d_shuffled = np.pad(d_shuffled, ((0, 0), (0, 0), (0, 10 - n_grasps), (0, 0)))
            else:
                d_shuffled = np.pad(d_shuffled, ((0, 0), (0, 0), (0, 10 - n_grasps), (0, 0)))
                '''pad_array = np.empty((len(d_shuffled), 1, 10-n_grasps, 19))
                for batch in range(len(d_shuffled)):
                    padding_idx = np.random.randint(n_grasps, size=10-n_grasps)
                    for idx in range(len(padding_idx)):
                        padding_row = d_shuffled[batch, :, padding_idx[idx], :]
                        pad_array[batch, :, idx, :] = padding_row
                d_shuffled = np.append(d_shuffled, pad_array, axis=2)'''
            shuffled_data = np.append(shuffled_data, d_shuffled, axis=0)
            l_shuffled = [label] * d_shuffled.shape[0]
            shuffled_labels.extend(l_shuffled)
        shuffled_data = shuffled_data[1:, :, :, :]
        X = torch.Tensor(shuffled_data)  # torch.Tensor(input_data.reshape((-1, 1, n_grasps, 19)))
        y = shuffled_labels
        self.transform = transform
        if pre_sort:
            # allocate and shuffle the input data to prevent batches only being populated by a single object
            seed = torch.Generator(device='cpu')
            seed.manual_seed(42)
            idx = torch.randperm(X.size(0), generator=seed)
            x_train = X[idx, :, :, :]
            x_test = X[idx, :, :, :]
            y_train = []
            y_test = []
            for n in range(len(X)):
                y_train.append(y[idx[n]])
                y_test.append(y[idx[n]])
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, stratify=y, random_state=20
            )

        if train:
            data_max = x_train.view(-1, 19).max(
                0).values.flatten()  # Ollie: Take max from training before applying to testing
            data_min = x_train.view(-1, 19).min(
                0).values.flatten()
            self.data = (x_train - data_min) / (data_max - data_min)
            self.targets = y_train
            self.max_vals = data_max
            self.min_vals = data_min
        else:
            self.data = (x_test - data_min) / (data_max - data_min)
            self.targets = y_test

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obj_name = self.targets[idx]
        data_line = self.data[idx, :]
        sample = {'labels': obj_name, 'data': data_line}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


class GraspDataset(Dataset):

    def __init__(self, x_filename, y_filename, normalize=False, subset_indices=None):

        # load datasets
        self.X = torch.FloatTensor(np.load(x_filename).reshape(-1, 190))
        self.y_labels = np.load(y_filename).reshape(-1, 1)

        if normalize:
            self.X = (self.X - self.X.mean(dim=0))/self.X.std(dim=0)

        self.labels = np.unique(self.y_labels)
        self.label_to_cls = dict()
        self.cls_to_label = dict()
        self.label_indeces = dict()

        self.y = torch.zeros(self.y_labels.shape).long()
        for cls, label in enumerate(self.labels):
            self.y[self.y_labels == label] = cls

            self.label_to_cls[label] = cls
            self.cls_to_label[cls] = label
            self.label_indeces[label] = np.argwhere(self.y_labels == label)[:, 0]

        self.indices = subset_indices if subset_indices is not None else np.array(list(range(len(self.X))))

    def set_subset(self, indeces):
        assert max(indeces) < len(self.X), "incorrect subset found, index larger than dataset"
        self.indices = np.array(indeces)
        return True

    def get_indeces(self):
        return self.indices

    def get_indeces_by_label(self):
        return self.label_indeces

    def get_splits(self, train_ratio, valid_ratio):
        # the function splits this dataset into three datasets according to the ratios
        train_indices = []
        valid_indices = []
        test_indices = []
        for object_label in self.labels:
            object_indeces = self.label_indeces[object_label]
            random.shuffle(object_indeces)

            # split data indices in train, valid and test (not the actual data, just the indeces)
            idx_train = round(len(object_indeces) * train_ratio)
            idx_valid = round(len(object_indeces) * valid_ratio)
            train_indices += sorted(object_indeces[:idx_train])
            valid_indices += sorted(object_indeces[idx_train:idx_train + idx_valid])
            test_indices += sorted(object_indeces[idx_train + idx_valid:])

        random.shuffle(train_indices)
        random.shuffle(valid_indices)
        random.shuffle(test_indices)

        train_dataset = copy.deepcopy(self)
        train_dataset.set_subset(train_indices)
        print('Training data: {} datapoints'.format(int(len(train_dataset))))

        valid_dataset = copy.deepcopy(self)
        valid_dataset.set_subset(valid_indices)
        print('Training data: {} datapoints'.format(int(len(valid_dataset))))

        test_dataset = copy.deepcopy(self)
        test_dataset.set_subset(test_indices)
        print('Testing data: {} datapoints'.format(int(len(test_dataset))))

        return train_dataset, valid_dataset, test_dataset

    def decode_labels(self, clss):
        labels = []
        for cls in clss:
            labels.append(self.cls_to_label[cls])
        return labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.X[self.indices[idx]], self.y[self.indices[idx]], self.y_labels[self.indices[idx]]
