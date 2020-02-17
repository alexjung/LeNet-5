from parallenet import ParalleNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom
import onnx
from onnx import optimizer, utils

# Choose desired device for training
# device = torch.device("cpu")
device = torch.device("cuda")

try:
    viz = visdom.Visdom(raise_exceptions=True)
except ConnectionError:
    viz = None
    print("INFO: Could not connect to visdom...")

data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

net = ParalleNet()
net = net.to(device=device)

criterion = nn.CrossEntropyLoss()
criterion.to(device=device)
adam = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        adam.zero_grad()

        images = images.to(device=device)
        labels = labels.to(device=device)
        output = net(images)

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
        adam.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.to(device=device)
        labels = labels.to(device=device)
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1, 16):
        train_and_test(e)

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True, device=device)
    torch.onnx.export(net, dummy_input, "parallenet.onnx")

    onnx_model = onnx.load("parallenet.onnx")
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    main()
