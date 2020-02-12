"""Datasets for reduced atmospheric river and tropical cyclone detection dataset.
"""


import itertools
import os

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive

# pylint: disable=C0330


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


class ARTCTemporaldataset(ARTCDataset):
    """Dataset for reduced ARTC dataset with temporality functionality.
    """

    def __init__(
        self,
        path,
        sequence_length,
        prediction_shift=0,
        indices=None,
        transform_image=None,
        transform_labels=None,
        transform_sample=None,
        download=False,
    ):
        """Initialization. Sort by run and sort each run by date and time. The samples at the tendo of each run are invalid and are removed.
        The list is then flattened. Self.allowed contains the list of all valid indices. Self.files contains all indices for the construction
        of samples.

        Args:
            path (str): Path to the data or desired place the data will be downloaded to.
            indices (list): List of indices representing the subset of the data used for the current dataset.
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            download (bool): Flag to decide if data should be downloaded or not.
            temporality_length (int): The number of images used per sample.
        """
        super().__init__(path, indices, None, None, download)
        self.transform_image = transform_image
        self.transform_labels = transform_labels
        self.transform_sample = transform_sample
        self.sequence_length = sequence_length
        self.prediction_shift = prediction_shift
        sorted_by_run_and_date = [sorted(self.get_runs([i])) for i in [1, 2, 3, 4, 6]]
        self.allowed = list(itertools.chain(*[run[: -(self.sequence_length + self.prediction_shift)] for run in sorted_by_run_and_date]))
        self.files = list(itertools.chain(*sorted_by_run_and_date))

    def __len__(self):
        """Get length of dataset.

        Returns:
            int: Number of files contained in the dataset.
        """
        return len(self.allowed)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx (int): The index of the desired datapoint.

        Returns:
            obj, obj: The data and labels corresponding to the desired index. The type depends on the applied transforms.
        """
        sample = []
        idx = self.files.index(self.allowed[idx])
        for i in range(self.sequence_length):
            sample.append(np.load(os.path.join(self.path, self.files[idx + i])))
        data = [image["data"] for image in sample]
        if self.prediction_shift > 0:
            target = np.load(os.path.join(self.path, self.files[idx + i + self.prediction_shift]))
            labels = target["labels"]
        else:
            labels = sample[-1]["labels"]
        if self.transform_image:
            for i, image in enumerate(data):
                data[i] = self.transform_image(image)
        if self.transform_labels:
            labels = self.transform_labels(labels)
        if self.transform_sample:
            data = self.transform_sample(data)
        return data, labels
