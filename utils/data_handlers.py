from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch

class ObjectGraspsDataset(Dataset):
    """Object Grasps Dataset"""

    def __init__(self, data_array_file, labels_file, transform=None, train: bool = True):
        """
        Args:
            array_file (string): Path to the saved ndarray file with grasp raw data.
            labels_file (string): Path to the saved ndarray file with labels.
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        input_data = np.load(data_array_file)
        input_targets = np.load(labels_file)
        # reshape data to align sensors
        X = torch.Tensor(input_data.reshape((-1, 1, 10, 19)))
        y = input_targets
        self.transform = transform
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=20
        )
        data_max = X_train.view(-1, 19).max(0).values.flatten()  # Luca: this finds the max for each sensor separately (they might have different baselines)
        data_min = X_train.view(-1, 19).min(0).values.flatten()  # the min in the normalization takes care of the negatives already (we push the whole distribution up)
        if train:
            self.data = (X_train-data_min)/(data_max-data_min)
            self.targets = y_train
        else:
            self.data = (X_test-data_min)/(data_max-data_min)
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
