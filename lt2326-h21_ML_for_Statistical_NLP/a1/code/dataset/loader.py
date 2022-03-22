import h5py

import numpy as np
import torch
from torch.utils import data


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path: str):
        super().__init__()

        with h5py.File(file_path) as h5_file:
            images_nm, labels_nm = h5_file.keys()
            self.images = np.array(h5_file[images_nm])
            self.labels = np.array(h5_file[labels_nm])
            self.mean_stddev = h5_file.attrs["MeanDev"]

    def __getitem__(self, index):
        # get data
        x = self.images[index]
        x = torch.as_tensor(x, dtype=torch.float)

        # get label
        y = self.labels[index]
        y = torch.as_tensor(y, dtype=torch.long)
        return (x, y)

    def __len__(self):
        return self.images.shape[0]


if __name__ == "__main__":
    train = HDF5Dataset("/scratch/guszarzmo/lt2326_labs/lab1/data/train.h5")

    num_epochs = 2
    loader_params = {"batch_size": 100, "shuffle": True, "num_workers": 6}
    data_loader = data.DataLoader(train, **loader_params)

    for i in range(num_epochs):
        for j, (x, y) in enumerate(data_loader):
            print(j, x.shape, y.shape)
        print()
