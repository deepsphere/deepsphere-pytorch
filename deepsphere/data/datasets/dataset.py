"""Dataset for reduced atmospheric river and tropical cyclone detection dataset.
"""
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive


class ARTCDataset(Dataset):
    """Dataset for reduced atmospheric river and tropical cyclone dataset.
    """

    resource = "http://island.me.berkeley.edu/ugscnn/data/climate_sphere_l5.zip"

    def __init__(self, path, indices=None, transform_data=None, transform_labels=None, download=False):
        """Initialization.

        Args:
            path (str): Path to the data or desired place the data will be downloaded to.
            indices (list): List of indices representing the subset of the data used for the current dataset.
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            download (bool): Flag to decide if data should be downloaded or not.
        """
        self.path = path
        if download:
            self.download()
        self.files = indices if indices is not None else os.listdir(self.path)
        self.transform_data = transform_data
        self.transform_labels = transform_labels

    @property
    def indices(self):
        """Get files.

        Returns:
            list: List of strings, which represent the files contained in the dataset.
        """
        return self.files

    def __len__(self):
        """Get length of dataset.

        Returns:
            int: Number of files contained in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx (int): The index of the desired datapoint.

        Returns:
            obj, obj: The data and labels corresponding to the desired index. The type depends on the applied transforms.
        """
        item = np.load(os.path.join(self.path, self.files[idx]))
        data, labels = item["data"], item["labels"]
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_labels:
            labels = self.transform_labels(labels)
        return data, labels

    def get_runs(self, runs):
        """Get datapoints corresponding to specific runs.

        Args:
            runs (list): List of desired runs.

        Returns:
            list: List of strings, which represents the files in the dataset, which belong to one of the desired runs.
        """
        files = []
        for file in self.files:
            for i in runs:
                if file.endswith("{}-mesh.npz".format(i)):
                    files.append(file)
        return files

    def download(self):
        """Download the dataset if it doesn't already exist.
        """
        if not self.check_exists():
            download_and_extract_archive(self.resource, download_root=os.path.split(self.path)[0])
        else:
            print("Data already exists")

    def check_exists(self):
        """Check if dataset already exists.
        """
        return os.path.exists(self.path)
