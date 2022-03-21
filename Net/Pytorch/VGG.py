import torch
import torchvision
import numpy as np
import os
import sys

batch_size_train = 4
batch_size_test = 4
if not os.path.exists('files'):
    os.mkdir('files')
train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('files', train=True, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ])),
batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('files', train=False, download=True,
                         transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(
                             (0.1307,), (0.3081,))
                         ])),
batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, bias=False, padding='same')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, bias=False, padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, bias=False, padding='same')
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, bias=False, padding='same')
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, bias=False, padding='same')
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, bias=False, padding='same')

        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(F.dropout2d(self.conv1(x), 0.2))
        x = F.relu(F.dropout2d(self.conv2(x), 0.2))
        x = F.max_pool2d(x, 2)

        x = F.relu(F.dropout2d(self.conv3(x), 0.2))
        x = F.relu(F.dropout2d(self.conv4(x), 0.2))
        x = F.max_pool2d(x, 2)

        x = F.relu(F.dropout2d(self.conv5(x), 0.2))
        x = F.relu(F.dropout2d(self.conv6(x), 0.2))
        x = F.relu(F.dropout2d(self.conv7(x), 0.2))
        x = F.max_pool2d(x, 2)

        x = nn.Flatten()(x) # torch.Size([1, 1152])

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


x = torch.rand(size=(1, 1, 28, 28))
vgg = VGG()
def train(model, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for ind_epochs in range(epochs):
        print("epochs", str(ind_epochs))
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 1500 == 0:
                print(batch_idx / 1500)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()



def test(network):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


train(vgg, 5)
test(vgg)