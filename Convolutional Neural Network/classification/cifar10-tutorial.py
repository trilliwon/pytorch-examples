# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# for image
import matplotlib.pyplot as plt
import numpy as np


# Data

print('===> Loading Data...')
"""
- 데이터를 torch Tensor로 바꾸고 Normalization을 위해 transform.Compose 를 사용합니다.
- Compose 는 여러 transform 들을 chaining 합니다. 즉 여러 transform 진행합니다.
"""

transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
CIFAR10 training data를 ./data 에 다운로드 하고 tranfrom 을 진행합니다.
- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
- DataLoader 는 iterator를 반환합니다.
"""
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# test data 를 다운로드 합니다.
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


# 클래스들은 뭐, (날아라 비행기, 아우디 r8, 날아라 병아리 아니고 벌드, 귀여운 고양이, 등 10개 입니다.)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model

print('===> Building Model - ResNet18 - ...')
"""
- [resnet](https://arxiv.org/abs/1512.03385)
- Cross Entropy loss 함수를 사용합니다.
- stochastic gradient descent 를 사용합니다.
- [optim.SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)
"""

net = models.resnet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Test before training

"""
그냥 한번 테스트 
"""

print('===> Testing before training...')
# batch size 만큼 이미지를 가져옵니다.
data = iter(trainloader).next()

inputs, labels = data

# print(', '.join('%7s' % classes[labels[j]] for j in range(4)))
print('input size: ', inputs.size())

outputs = net(inputs)
loss = criterion(outputs, labels)

print('output size: ', outputs.size())
print('loss: ', loss.item())


"""
이미지 보기
"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()     # numpy vector로 바꾸기
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# [make_grid](https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid)
# imshow(torchvision.utils.make_grid(inputs))

print(', '.join('%7s' % classes[labels[j]] for j in range(4)))


# Training

"""
GPU 사용하기
"""

print('\n===> Training Start')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
if torch.cuda.device_count() > 1:
    print('\n===> Training on GPU!')
    net = nn.DataParallel(net)


epochs = 2 # dataset을 여러번 사용해 트레이닝을 시킵니다.

for epoch in range(epochs):
    print('\n===> epoch %d' % epoch)
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('\n===> Finished Training...')


# Test
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('\nAccuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('\nAccuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))