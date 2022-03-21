import torch
# import torchvision
# import numpy as np
# import os
# import sys

# batch_size_train = 4
# batch_size_test = 4
# if not os.path.exists('files'):
#     os.mkdir('files')
# train_loader = torch.utils.data.DataLoader(
# torchvision.datasets.MNIST('files', train=True, download=True,
#                          transform=torchvision.transforms.Compose([
#                            torchvision.transforms.ToTensor(),
#                            torchvision.transforms.Normalize(
#                              (0.1307,), (0.3081,))
#                          ])),
# batch_size=batch_size_train, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
# torchvision.datasets.MNIST('files', train=False, download=True,
#                          transform=torchvision.transforms.Compose([
#                            torchvision.transforms.ToTensor(),
#                            torchvision.transforms.Normalize(
#                              (0.1307,), (0.3081,))
#                          ])),
# batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
#


import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim


class VGG_ResNet(nn.Module):
    def __init__(self, nout=100):
        super(VGG_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.BN1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding='valid', bias=False)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='valid', bias=False)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='valid', bias=False)

        ## torch.Size([1, 64, 124, 124])
        # vgg 1
        self.conv7_1 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.BN2 = nn.BatchNorm2d(64)
        self.conv7_3 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding='valid', bias=False)
        # res 2
        self.conv8_1 = nn.Conv2d(64, 64, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='valid', bias=False)

        self.conv9_1 = nn.Conv2d(64, 128, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv9_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='valid', bias=False)
        ## torch.Size([1, 128, 58, 58])
        # vgg 3
        self.conv10_1 = nn.Conv2d(128, 128, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.conv10_2 = nn.Conv2d(128, 128, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.BN3 = nn.BatchNorm2d(128)
        self.conv10_3 = nn.Conv2d(128, 256, kernel_size=(5, 5), padding='valid', bias=False)
        # res 3
        self.conv11_1 = nn.Conv2d(128, 128, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv11_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='valid', bias=False)

        self.conv12_1 = nn.Conv2d(128, 256, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv12_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding='valid', bias=False)
        ## torch.Size([1, 256, 25, 25])
        # vgg 4
        self.conv13_1 = nn.Conv2d(256, 256, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.conv13_2 = nn.Conv2d(256, 256, kernel_size=(5, 5), padding='same', bias=False, padding_mode='zeros')
        self.BN4 = nn.BatchNorm2d(256)
        self.conv13_3 = nn.Conv2d(256, 512, kernel_size=(5, 5), padding='valid', bias=False)
        # res 4
        self.conv14_1 = nn.Conv2d(256, 256, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv14_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding='valid', bias=False)

        self.conv15_1 = nn.Conv2d(256, 512, kernel_size=(2, 2), padding='same', bias=False, padding_mode='zeros')
        self.conv15_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding='valid', bias=False)
        ## torch.Size([1, 512, 8, 8])
        self.fc1 = nn.Linear(8192, 2048, bias=False)
        self.fc2 = nn.Linear(2048, nout, bias=False)


    def forward(self, x):
        # vgg 1
        x = F.relu(F.dropout2d(self.conv1(x), 0.2))
        x = F.relu(F.dropout2d(self.BN1(self.conv2(x)), 0.2))
        x = F.max_pool2d(x, 2)
        # res block 1
        y = F.dropout2d(F.relu(self.conv3(x)), 0.2)
        y = F.dropout2d(F.relu(self.conv4(y)), 0.2)

        y = F.dropout2d(F.relu(self.conv5(y)), 0.2)
        y = F.relu(self.conv6(y))
        # com 1
        x = F.relu(self.conv2_2(x))
        x = x + y
        x = F.relu(x)

        # vgg 2
        x = F.relu(F.dropout2d(self.conv7_1(x), 0.2))
        x = F.relu(F.dropout2d(self.BN2(self.conv7_2(x)), 0.2))
        x = F.max_pool2d(x, 2)
        # res block 2
        y = F.dropout2d(F.relu(self.conv8_1(x)), 0.2)
        y = F.dropout2d(F.relu(self.conv8_2(y)), 0.2)

        y = F.dropout2d(F.relu(self.conv9_1(y)), 0.2)
        y = F.relu(self.conv9_2(y))

        # com 2
        x = F.relu(self.conv7_3(x))

        x = x + y
        x = F.relu(x)

        # vgg 3
        x = F.relu(F.dropout2d(self.conv10_1(x), 0.2))
        x = F.relu(F.dropout2d(self.BN3(self.conv10_2(x)), 0.2))
        x = F.max_pool2d(x, 2)
        # res block 3
        y = F.dropout2d(F.relu(self.conv11_1(x)), 0.2)
        y = F.dropout2d(F.relu(self.conv11_2(y)), 0.2)

        y = F.dropout2d(F.relu(self.conv12_1(y)), 0.2)
        y = F.relu(self.conv12_2(y))
        # com 3
        x = F.relu(self.conv10_3(x))

        x = x + y
        x = F.relu(x)

        # vgg 4
        x = F.relu(F.dropout2d(self.conv13_1(x), 0.2))
        x = F.relu(F.dropout2d(self.BN4(self.conv13_2(x)), 0.2))
        x = F.max_pool2d(x, 2)
        # res block 4
        y = F.dropout2d(F.relu(self.conv14_1(x)), 0.2)
        y = F.dropout2d(F.relu(self.conv14_2(y)), 0.2)

        y = F.dropout2d(F.relu(self.conv15_1(y)), 0.2)
        y = F.relu(self.conv15_2(y))
        # com 4
        x = F.relu(self.conv13_3(x))

        x = x + y
        x = F.relu(x)

        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = torch.flatten(x)
        x = F.dropout(F.relu(self.fc1(x)), 0.2)
        x = self.fc2(x)
        return F.softmax(x)


# def train(model, epochs=10):
#     model.train()
#     optimizer = optim.Adam(model.parameters(), lr=0.1)
#     for ind_epochs in range(epochs):
#         print("epochs", str(ind_epochs))
#         for batch_idx, (data, target) in enumerate(train_loader):
#             if batch_idx % 1500 == 0:
#                 print(batch_idx / 1500)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#
#
#
# def test(network):
#   network.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_loader:
#       output = network(data)
#       test_loss += F.nll_loss(output, target, size_average=False).item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= len(test_loader.dataset)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))



X_test_tensor = torch.rand(size=(1, 3, 256, 256))
vg_re = VGG_ResNet()
a = vg_re(X_test_tensor)
print(a.size())

