import os
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from tqdm.auto import tqdm
import random


class ShapeNetSemSeg(Dataset):
    def __init__(
        self, root_path, categories, split="train", transform=None, num_subsample=2048
    ):
        super(ShapeNetSemSeg, self).__init__()
        self.root_path = root_path
        self.categories = categories  # Should be a list of categories
        self.split = split
        self.transform = transform
        self.num_subsample = num_subsample  # Number of points to subsample to

        self.files = []
        self.label_map = {}
        self.num_classes = 0
        self.load_data_paths()

        self.pointclouds = []
        self.load()

    def load_data_paths(self):
        all_labels = []
        for category in self.categories:
            split_file_path = os.path.join(
                self.root_path, category, f"{self.split}_files.txt"
            )
            if not os.path.isfile(split_file_path):
                raise ValueError(f"Split file {split_file_path} does not exist")
            with open(split_file_path, "r") as file:
                file_paths = [
                    os.path.join(self.root_path, category, line.strip())
                    for line in file.readlines()
                ]
                self.files.extend(file_paths)
                for fp in file_paths:
                    with h5py.File(fp, "r") as h5_file:
                        labels = h5_file["label_seg"][:]
                        all_labels.extend(np.unique(labels))

        unique_labels = np.unique(all_labels)
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

    def load(self):
        for filepath in tqdm(self.files, desc=f"Loading {self.split} data"):
            with h5py.File(filepath, "r") as h5_file:
                points = h5_file["data"][:]  # shape (1024, 10000, 3)
                labels = h5_file["label_seg"][:]  # shape (1024, 10000)

                # Print shape of numpy points
                print(points.shape)

                # Normalize the point cloud
                # Calculating the mean across all point clouds and all points
                overall_mean = points.mean(axis=(0, 1)).reshape(
                    1, 3
                )  # Averages across both the point clouds and the points

                # Calculate the mean of the standard deviations across x, y, z
                overall_std = points.std(axis=(0, 1)).mean().reshape(1, 1)

                points = (points - overall_mean) / overall_std

                # Process each point cloud individually
                for i in range(points.shape[0]):
                    single_points = points[i]  # (10000, 3)
                    single_labels = labels[i]  # (10000,)

                    # Map labels to sequential integers starting from 0
                    mapped_labels = np.vectorize(self.label_map.get)(single_labels)
                    mapped_labels = mapped_labels.reshape(
                        -1, 1
                    )  # Reshapes labels to (10000, 1)

                    # Concatenate points with their labels to make each point's shape (4,)
                    point_labels = np.hstack((single_points, mapped_labels))

                    self.pointclouds.append(
                        torch.tensor(point_labels, dtype=torch.float32)
                    )

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        pointcloud_with_labels = self.pointclouds[idx]
        # Random subsampling
        indices = np.random.choice(
            pointcloud_with_labels.shape[0], self.num_subsample, replace=False
        )
        subsampled_pointcloud_with_labels = pointcloud_with_labels[indices]

        sample = {"pointcloud": subsampled_pointcloud_with_labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
