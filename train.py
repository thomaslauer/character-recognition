from emnist_dataset import EmnistDataset

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

plt.ion()

class Net(nn.Module):
    def __init__(self, classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Net2(nn.Module):
    def __init__(self, classes):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5, 1)
        self.conv2 = nn.Conv2d(100, 100, 5, 1)
        self.fc1 = nn.Linear(4*4*100, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*100)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_densenet121(num_classes):
    model = torchvision.models.densenet121(pretrained=True)

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # define fully connected layer
    fc = nn.Sequential(
        nn.Linear(1024, 460),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(460, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = fc

    return model


def load_vgg16(num_classes):
    model = torchvision.models.vgg16(pretrained=True)

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # define fully connected layer
    fc = nn.Sequential(
        nn.Linear(25088, 460),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(460, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = fc

    return model

def load_vgg11(num_classes):
    model = torchvision.models.vgg11(pretrained=True)

    # freeze weights
    for param in model.parameters():
        param.requires_grad = False

    # define fully connected layer
    fc = nn.Sequential(
        nn.Linear(25088, 460),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(460, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = fc

    return model


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    test_loss = 0
    correct = 0

    full_target = None
    full_pred = None

    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if not isinstance(full_target, np.ndarray):
                full_target = target.cpu().numpy()
                full_pred = pred.view_as(target).cpu().numpy()
            else: 
                full_target = np.concatenate((full_target, target.cpu().numpy()))
                full_pred = np.concatenate((full_pred, pred.view_as(target).cpu().numpy()))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    confusion_mat = confusion_matrix(full_target, full_pred)
    return correct / len(test_loader.dataset), confusion_mat


def main():

    # All these args were stolen from the pytorch mnist example
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    split = 'bymerge'

    # Prep the emnist dataset
    emnist = EmnistDataset()
    emnist.prep_data()

    """
    # Get the tensors from the dataset
    train_dataset = emnist.load_split(split, 'train')

    test_dataset = emnist.load_split(split, 'test')
    """

    """
    # EMNIST dataset loader
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join("../output", split, "train"),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join("../output", split, "test"),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    """

    full_dataset = torchvision.datasets.ImageFolder(
        root="../chars74k/English/Fnt",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomAffine((0, 360)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.test_batch_size,
        num_workers=4,
        shuffle=True
    )

    num_classes = 62
    model = load_vgg16(num_classes).to(device)
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=args.momentum)

    if not os.path.exists('progress.csv'):
        with open('progress.csv', 'a') as progress_file:
            progress_file.write('epoch,test_loss\n')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, confusion_mat = test(args, model, device, test_loader)

        """
        plt.clf()
        plt.cla()
        plt.imshow(confusion_mat)
        for (i,j),label in np.ndenumerate(confusion_mat):
            plt.gca().text(i,j,label, ha='center', va='center')
        plt.show()
        plt.pause(0.001)
        """

        with open('progress.csv', 'a') as progress_file:
            progress_file.write('{},{}\n'.format(epoch, test_loss))
    
    if args.save_model:
        torch.save(model.state_dict(), 'emnist_model.pt')

if __name__ == '__main__':
    main()

