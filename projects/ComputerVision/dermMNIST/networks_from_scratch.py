import numpy as np
from sklearn import metrics
import torch
from matplotlib import pyplot as plt
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchmetrics import Accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    """A basic convolutional neural network."""

    def __init__(self, im_width, im_height, num_classes: int = 7, input_channels=3, cnn_start_channels=16,
                 preclassifier_channels=10240):
        super().__init__()

        self.backbone_channels = cnn_start_channels
        self.im_width = im_width
        self.im_height = im_height

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=self.backbone_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.backbone_channels = self.backbone_channels * 2
        self.im_width = self.im_width // 2
        self.im_height = self.im_height // 2
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_start_channels, out_channels=self.backbone_channels, kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.backbone2 = nn.Sequential(
            nn.Conv2d(in_channels=self.backbone_channels, out_channels=self.backbone_channels, kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.backbone3 = nn.Sequential(
            nn.Conv2d(in_channels=self.backbone_channels, out_channels=self.backbone_channels, kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.backbone4 = nn.Sequential(
            nn.Conv2d(in_channels=self.backbone_channels, out_channels=self.backbone_channels, kernel_size=3,
                      padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_channels * self.im_width * self.im_height, preclassifier_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(preclassifier_channels, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.backbone3(x)
        x = self.backbone4(x)
        x = x.view(-1, self.backbone_channels * self.im_width * self.im_height)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=7, input_channels=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Trainer:
    def __init__(self, train_loader, val_loader, model, num_epochs=100, num_classes=7):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.accuracy_metric = Accuracy(task="multiclass", average="micro", num_classes=int(num_classes)).to(device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-5)
        self.best_acc = 0

    def train_val(self):
        # Initialize accuracy metric
        for epoch in range(self.num_epochs):
            train_acc, train_loss = self.train(epoch)

            eval_acc, eval_loss = self.validation()

            print(
                f"[{epoch + 1}/{self.num_epochs}] "
                f"train_loss: {train_loss:.3f} - "
                f"train_acc: {train_acc:.3f} - "
                f"eval_loss: {eval_loss:.3f} - "
                f"eval_acc: {eval_acc:.3f}"
            )

            print()

    def train(self, epoch):
        # Training loop
        if epoch % 50 == 0:
            for g in self.optimizer.param_groups:
                g['lr'] = 0.001
        self.model.train()
        running_loss = 0.0
        for images, labels in self.train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1)
            labels = labels.type(torch.LongTensor).to(device)  # Was previously handled in loader

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backprop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update running loss
            running_loss += loss.item()

            # Update accuracy for train data
            _, predicted = torch.max(outputs, 1)
            self.accuracy_metric.update(predicted, labels)
        # Calculate train loss and accuracy
        train_loss = running_loss / len(self.train_loader)
        train_acc = self.accuracy_metric.compute()
        self.accuracy_metric.reset()
        return train_acc, train_loss

    def validation(self):
        # Evaluation loop
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            pred_class = list()
            actual_class = list()
            for images, labels in self.val_loader:
                images = images.to(device)
                labels = labels.to(device).view(-1)
                labels = labels.type(torch.LongTensor).to(device)  # Was previously handled in loader

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Update running loss
                running_loss += loss.item()

                # Update accuracy for eval data
                _, predicted = torch.max(outputs, 1)

                self.accuracy_metric.update(predicted, labels)

        # print(confusion_matrix)
        # Calculate eval loss and accuracy
        eval_loss = running_loss / len(self.val_loader)
        eval_acc = self.accuracy_metric.compute()
        self.accuracy_metric.reset()
        confusion_matrix = metrics.confusion_matrix(actual_class, pred_class)
        print(confusion_matrix)

        if eval_acc > self.best_acc:
            self.best_acc = eval_acc
            torch.save(self.model.state_dict(), r"D:\Models\CV\DERMAMNIST\params.pth")
        return eval_acc, eval_loss


def show_examples(dataset: Dataset, k: int = 5):
    """Plots a row of k random examples and their label from a dataset."""
    rand_list = np.random.choice(len(dataset) - 1, size=k, replace=False)

    fig, axes = plt.subplots(1, k, figsize=(10, 2))  # Initialize a figure
    for ax_ind, rand_ind in enumerate(rand_list):
        plt.sca(axes[ax_ind])
        image = np.swapaxes(dataset[rand_ind][0], 0, 2)
        plt.imshow(image)
        axes[ax_ind].title.set_text(dataset[rand_ind][1])
    plt.show()  # Plot the figure


def count_classes_check_imbalance(dataset, access_level=0):
    labels_count = dict()
    for i in range(len(dataset)):
        if access_level == 1:
            label = dataset[i][1][0]
        else:
            label = dataset[i][1]
        # print(train_dataset[i][1][0])
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

    num_classes = int(max(list(labels_count.keys())) + 1)
    return num_classes