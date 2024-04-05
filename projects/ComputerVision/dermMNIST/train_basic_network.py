from medmnist import DermaMNIST

import torch
from torch.nn import Linear

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from networks_from_scratch import ConvNet, Trainer, show_examples, \
    count_classes_check_imbalance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IM_WIDTH = 28
IM_HEIGHT = 28
make_plots = False
model_type = 'basic'
backbone_channels = 128
pretrained = True
# Best setting
# model_type = 'resnet50'
# pretrained = True


# Basic CCN/FCN mix achieves 0.69 accuracy
# resnet50 accuracy published in nature is only 73.1%! https://www.nature.com/articles/s41597-022-01721-8/tables/4


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
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    # Initialize model
    # model = ConvNet(im_width=IM_WIDTH, im_height=IM_HEIGHT, num_classes=7)
    if model_type == 'basic':
        model = ConvNet(im_width=im_width, im_height=im_height, num_classes=num_classes, input_channels=input_channels,
                        cnn_start_channels=128, preclassifier_channels=10240)
    elif model_type == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = Linear(in_features=2048, out_features=7)
    else:
        raise NotImplementedError()

    model = model.to(device)

    num_epochs = 100
    trainer = Trainer(train_loader=train_loader, val_loader=val_loader, model=model, num_epochs=num_epochs,
                      num_classes=num_classes)
    trainer.train_val()

    # Achieves 0.69 accuracy but does not increase further in either training or validation sets.
    # Investigate underfit vs data issue?


if __name__ == '__main__':
    main()
