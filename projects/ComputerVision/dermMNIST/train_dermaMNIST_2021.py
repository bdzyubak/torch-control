import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms


from networks_from_scratch import BasicNet, train_val, show_examples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IM_WIDTH = 28
IM_HEIGHT = 28
make_plots = False


def main():
    # Evaluate training in 2021 version of the data
    # https://zenodo.org/record/4269852/files/dermamnist.npz?download=1

    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    data = np.load(r"D:\data\CV\dermamnist_2021.npz")
    train_dataset = DERMAMNIST2021_Dataset(data['train_images'], data['train_labels'], transform=data_transforms)
    val_dataset = DERMAMNIST2021_Dataset(data['val_images'], data['val_labels'], transform=data_transforms)

    num_classes = _count_classes_check_imbalance(data['train_labels'], access_level=1)
    input_channels = train_dataset[0][0].shape[0]
    im_width = train_dataset[0][0].shape[1]
    im_height = train_dataset[0][0].shape[2]

    if make_plots:
        show_examples(train_dataset)

    # # Initialize training and evaluation loaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    # Initialize model
    # model = ConvNet(im_width=IM_WIDTH, im_height=IM_HEIGHT, num_classes=7)
    model = BasicNet(im_width=im_width, im_height=im_height, num_classes=num_classes, input_channels=input_channels)
    model = model.to(device)

    num_epochs = 40
    train_val(train_loader=train_loader, eval_loader=val_loader, model=model, num_epochs=num_epochs, num_classes=num_classes)

    # Result: achieves 0.69 accuracy but does not increase past this point


class DERMAMNIST2021_Dataset(Dataset):
    def __init__(self, images_np, labels_np, transform=None, target_transform=None):
        self.images_np = images_np
        self.labels_np = labels_np
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels_np)

    def __getitem__(self, idx):
        image = np.swapaxes(self.images_np[idx], 2, 0)
        label = self.labels_np[idx][0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def _count_classes_check_imbalance(labels, access_level=0):
    labels_count = dict()
    for temp in labels:
        label = temp[0] if access_level else temp
        if label in labels_count:
            labels_count[label] += 1
        else:
            labels_count[label] = 1

    print(labels_count)
    total_counts = np.sum(list(labels_count.values()))
    keys = list(labels_count.keys())
    keys.sort()
    labels_count_ratio = {key: round((labels_count[key] / total_counts) * 100, 2) for key in keys}
    print(f"The label composition in % is: {labels_count_ratio}")

    num_classes = np.max(labels) + 1
    return num_classes


if __name__ == '__main__':
    main()
