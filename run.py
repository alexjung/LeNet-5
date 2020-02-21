import argparse
from typing import Text

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx
from onnx import optimizer, utils

from parallenet import ParalleNet
from mininception import MinInception

supported_models = {
    "parallenet": ParalleNet(),
    "mininception": MinInception()
}
model_shapes = {
    "parallenet": (1, 1, 28, 28),
    "mininception": (1, 1, 28, 28)
}

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}

parser = argparse.ArgumentParser(
    description="Tool to train simple models (ParalleNet, MinInception) for research purposes."
)
parser.add_argument(
    "model",
    type=Text,
    default="parallenet",
    choices=["parallenet", "mininception"],
    help="Choose model to train."
)
parser.add_argument(
    "device",
    type=Text,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Choose the device training should be run on."
)


def main():
    args = parser.parse_args()

    device = torch.device(args.device)
    model_name = args.model

    try:
        viz = visdom.Visdom(raise_exceptions=True)
    except ConnectionError:
        viz = None
        print("INFO: Could not connect to visdom...")

    net = supported_models[model_name]
    net = net.to(device=device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)
    adam = optim.Adam(net.parameters(), lr=2e-3)

    if model_name in ["parallenet", "mininception"]:

        data_train = MNIST('./data/mnist',
                           download=True,
                           transform=transforms.Compose([
                               transforms.Resize((28, 28)),
                               transforms.ToTensor()]))
        data_test = MNIST('./data/mnist',
                          train=False,
                          download=True,
                          transform=transforms.Compose([
                              transforms.Resize((28, 28)),
                              transforms.ToTensor()]))

        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
        data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

    else:
        print("ERROR: Not supported yet")
        exit(1)

    for epoch in range(1, 16):
        train_and_test(data_train_loader, data_test_loader, net, criterion, adam, epoch, device, viz)

    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    torch.onnx.export(net, dummy_input, "{}.onnx".format(args.model))

    onnx_model = onnx.load("{}.onnx".format(args.model))
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)


def train(train_loader, model, criterion, optimizer, epoch, device, viz):
    global cur_batch_win
    model.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device=device)
        labels = labels.to(device=device)
        output = model(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().item()))

        # Update Visualization
        if viz and viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test(test_loader, model, criterion, device):
    model.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        output = model(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(test_loader.dataset)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().item(), float(total_correct) / len(test_loader.dataset)))


def train_and_test(train_loader, test_loader, model, criterion, optimizer, epoch, device, viz):
    train(train_loader, model, criterion, optimizer, epoch, device, viz)
    test(test_loader, model, criterion, device)


if __name__ == '__main__':
    main()
