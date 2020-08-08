from utils.data_augmentation import transform_train, transform_test
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class OceanDataset(Dataset):
    def __init__(self, root, input_data, aug):
        self.file_data = input_data['FileID'].values
        self.label_data = input_data['SpeciesID'].values if 'SpeciesID' in input_data.columns else None
        self.aug = aug

        self.img_data = [str(root+i+'.jpg') for i in self.file_data]

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img = self.img_data[index]
        img = Image.open(img)

        if self.aug is not None:
            img = self.aug(img)

        if self.label_data is not None:
            return img, self.file_data[index], self.label_data[index]
        else:
            return img, self.file_data[index]


def get_train_dataloader(img_path, train_csv, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        img_path: the root directory of images
        train_csv: training set's labels loaded by pandas
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader: torch dataloader object
    """

    train_data = OceanDataset(root=img_path, input_data=train_csv, aug=transform_train)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_data_loader

def get_val_dataloader(img_path, val_csv, batch_size=16, num_workers=2, shuffle=False):
    """ return Validation dataloader
    Args:
        img_path: the root directory of images
        val_csv: validation set's labels loaded by pandas
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: val_data_loader: torch dataloader object
    """

    val_data = OceanDataset(root=img_path, input_data=val_csv, aug=transform_test)
    val_data_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return val_data_loader

def get_test_dataloader(img_path, test_csv, batch_size=16, num_workers=2, shuffle=False):
    """ return Testing dataloader
    Args:
        img_path: the root directory of images
        test_csv: testing set's labels loaded by pandas
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_data_loader: torch dataloader object
    """


    test_data = OceanDataset(root=img_path, input_data=test_csv, aug=transform_test)
    test_data_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return test_data_loader