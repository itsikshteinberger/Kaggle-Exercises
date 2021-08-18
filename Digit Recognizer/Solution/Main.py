from matplotlib import pyplot as plt
from torch import nn
from torchvision import models

from Dataset import DigitSet
import torch


from Model1 import Net2, Net3
from config import *



def showData(image):
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.show()


def main():
    trainset = DigitSet(root_dir='data', csv_file='train.csv')
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = DigitSet(root_dir='data', csv_file='test.csv', labels_file='result.csv')
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    l,i = trainset[0]
    i = i.reshape((1,1,28,28))


    model = Net3()
    print(model((i).float()).shape)



if __name__ == "__main__":
    main()
