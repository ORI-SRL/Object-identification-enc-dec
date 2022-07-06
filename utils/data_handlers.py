from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch


class ObjectGraspsDataset(Dataset):
    """Object Grasps Dataset"""

    def __init__(self, data_array_file, labels_file, n_grasps, data_max=None, data_min=None, transform=None,
                 train: bool = True, pre_sort=False):
        """
        Args:
            array_file (string): Path to the saved ndarray file with grasp raw data.
            labels_file (string): Path to the saved ndarray file with labels.
            n_grasps (int): number of grasps to be trained on for data reshuffle
        """

        input_data = np.load(data_array_file)
        input_targets = np.load(labels_file)
        shuffled_data = np.empty((1, 1, 10, 19))
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
            d_shuffled = np.pad(d_shuffled, ((0, 0), (0, 0), (0, 10 - n_grasps), (0, 0)))
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
