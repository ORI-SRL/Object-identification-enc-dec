from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch


class ObjectGraspsDataset(Dataset):
    """Object Grasps Dataset"""

    def __init__(self, data_array_file, labels_file, n_grasps, transform=None,
                 train: bool = True):
        """
        Args:
            array_file (string): Path to the saved ndarray file with grasp raw data.
            labels_file (string): Path to the saved ndarray file with labels.
            n_grasps (int): number of grasps to be trained on for data reshuffle
        """

        input_data = np.load(data_array_file)
        input_targets = np.load(labels_file)
        shuffled_data = np.empty((1, 1, n_grasps, 19))
        shuffled_labels = []
        labels = list(set(input_targets))
        # reshape data to align sensors but maintain label for each grasp
        for label in labels:
            d_label = input_data[label == input_targets]
            d_shuffled = d_label.reshape((-1, 1, n_grasps, 19))
            shuffled_data = np.append(shuffled_data, d_shuffled, axis=0)
            l_shuffled = [label] * d_shuffled.shape[0]
            shuffled_labels.extend(l_shuffled)
        shuffled_data = shuffled_data[1:, :, :, :]
        X = torch.Tensor(shuffled_data)  # torch.Tensor(input_data.reshape((-1, 1, n_grasps, 19)))
        y = shuffled_labels
        self.transform = transform
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=20
        )
        data_max = X_train.view(-1, 19).max(
            0).values.flatten()  # Luca: this finds the max for each sensor separately (they might have different baselines)
        data_min = X_train.view(-1, 19).min(
            0).values.flatten()  # the min in the normalization takes care of the negatives already (we push the whole distribution up)
        if train:
            self.data = (X_train - data_min) / (data_max - data_min)
            self.targets = y_train
        else:
            self.data = (X_test - data_min) / (data_max - data_min)
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
