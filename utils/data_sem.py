import torch
from torch.utils.data import DataLoader, random_split
from .sem_dataset import ShapeNetSemSeg


def get_train_val_test_datasets(dataset, train_ratio, val_ratio):
    assert (
        train_ratio + val_ratio
    ) <= 1, "Sum of train_ratio and val_ratio should not exceed 1."
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size

    # Randomly split the dataset into train, validation, and test datasets
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_set, val_set, test_set


def get_train_val_test_loaders(
    root_path,
    category,
    train_ratio,
    val_ratio,
    train_batch_size,
    val_test_batch_size,
    num_workers=0,
):
    # Instantiate datasets for the train, validation, and test splits
    train_dataset = ShapeNetSemSeg(
        root_path=root_path, category=category, split="train"
    )
    val_dataset = ShapeNetSemSeg(root_path=root_path, category=category, split="val")
    test_dataset = ShapeNetSemSeg(root_path=root_path, category=category, split="test")

    # Create DataLoaders for each dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_data_iterator(iterable):
    """Create an infinite iterator to loop over a DataLoader."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)  # Refresh the iterator if it runs out of data
