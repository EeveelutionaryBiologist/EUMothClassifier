
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data and sets up a train/test split
"""

import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path


# Fraction of the test data split
test_split = 0.1

# Check available CPUs for data loader
CPU_COUNT = os.cpu_count()

if CPU_COUNT > 8:
    NUM_WORKERS = 8     # We experience diminishing returns for more...
else:
    NUM_WORKERS = max((CPU_COUNT-1, 1))


# Batch size & image crop dependent on available memory 
total_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
total_free_gpu_memory_gb = round(total_free_gpu_memory * 1e-9, 3)

if total_free_gpu_memory_gb >= 16:
  BATCH_SIZE = 256
  IMAGE_SIZE = 224
  print(f"GPU memory available is {total_free_gpu_memory_gb} GB, using batch size of {BATCH_SIZE} and image size {IMAGE_SIZE}")
else:
  BATCH_SIZE = 128
  IMAGE_SIZE = 224
  print(f"GPU memory available is {total_free_gpu_memory_gb} GB, using batch size of {BATCH_SIZE} and image size {IMAGE_SIZE}")



def split_list_by_fraction(entries: list, test_fraction: float=0.2):
    """
    Randomly split a list into two sublists by a given fraction.
    """
    # Create a copy and shuffle it
    random.shuffle(entries)

    # Calculate split point
    split_point = int(len(entries) * (1 -test_fraction))

    # Split the list
    list_1 = entries[:split_point]
    list_2 = entries[split_point:]

    return list_1, list_2


def create_dataloaders(
    data_dir: str,
    transform: transforms.Compose,
    batch_size: int=BATCH_SIZE,
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    data_dir: directory name with folders containing training examples.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
  """
  # Check if directory exists
  if not os.path.exists(data_dir):
      raise FileNotFoundError(f"Training directory not found at: {data_dir}")

  # Update transform according to memory:
  transforms.crop_size = IMAGE_SIZE
  transforms.resize_size = IMAGE_SIZE

  try:
      # Use ImageFolder to create dataset(s)
      image_data = datasets.ImageFolder(data_dir, transform=transform)

  except Exception as e:
      raise Exception(f"Error loading image data from directories: {str(e)}")

  # Get class names
  class_names = image_data.classes

  # Do a train-test split
  set_1, set_2 = split_list_by_fraction(entries=list(range(0, len(image_data))), test_fraction=test_split)
  datasplit_1 = torch.utils.data.Subset(image_data, set_1)
  datasplit_2 = torch.utils.data.Subset(image_data, set_2)

  # Turn images into data loaders
  train_dataloader = DataLoader(
      datasplit_1,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      datasplit_2,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

