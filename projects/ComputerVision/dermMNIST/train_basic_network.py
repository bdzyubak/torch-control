from medmnist import DermaMNIST

import torch

from torch.utils.data import DataLoader
from torchvision import transforms


from networks_from_scratch import ConvNet, train_val, show_examples, \
    count_classes_check_imbalance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IM_WIDTH = 28
IM_HEIGHT = 28
make_plots = False


def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load training and evaluation splits
    train_dataset = DermaMNIST(split='train', transform=train_transforms, download=True)
    val_dataset = DermaMNIST(split='val', transform=train_transforms, download=True)

    if make_plots:
        show_examples(train_dataset)

    num_classes = count_classes_check_imbalance(train_dataset, access_level=1)
    input_channels = train_dataset[0][0].shape[0]
    im_width = train_dataset[0][0].shape[1]
    im_height = train_dataset[0][0].shape[2]

    # Initialize training and evaluation loaders
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    eval_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    # Initialize model
    # model = ConvNet(im_width=IM_WIDTH, im_height=IM_HEIGHT, num_classes=7)
    model = ConvNet(im_width=im_width, im_height=im_height, num_classes=num_classes, input_channels=input_channels)
    model = model.to(device)

    num_epochs = 40
    train_val(train_loader=train_loader, eval_loader=eval_loader, model=model, num_epochs=num_epochs, num_classes=num_classes)


if __name__ == '__main__':
    main()
