from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import __version__
from torchvision import transforms
from utils.progressBar import printProgressBar

import utils.medicalDataLoader
import argparse
from utils.my_utils import *

import model.medformer as mdf
import model.medformer_utils
import random
import torch
import pdb
from torch import optim
from dice.my_dice import *

import warnings
warnings.filterwarnings("ignore")


def getTargetSegmentation2(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze(1)



n=0
q=0
s=0
dice=[]


# Load the saved model

net = mdf.MedFormer(1, 4)
checkpoint = torch.load('<saved_model_path>')
net.load_state_dict(checkpoint)


# net = mdf.MedFormer(1, 4)

Dice_loss = DiceLoss(classes=[1,2,3], ignore_index=0)

# Set the model to evaluation mode
net.eval()

# Define your validation data loader
root_dir = './data/'

print(' Dataset: {} '.format(root_dir))

## DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION
    
transform = transforms.Compose([
        transforms.ToTensor()
    ])

mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])


val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

val_loader = DataLoader(val_set,
                            worker_init_fn=np.random.seed(0),
                            num_workers=0,
                            shuffle=False)


# Define variables for computing accuracy
correct = 0
test_loss = 0
total_pixels = 0
verbose = False


with torch.no_grad():
    n=0
    j=0
    q=0
    for data in val_loader:
        images, labels, img_names = data[1], data[2], data[3]

        ### From numpy to torch variables
        labels = to_var(labels)
        images = to_var(images)
        labels_long = labels.type(torch.LongTensor)
        segmentation_classes = getTargetSegmentation(labels)
        pred = net(images)
        #print(pred.size())
        m = nn.Softmax(dim=1)
        pred = m(pred)
        p = pred.argmax(dim=1, keepdim=True)
        pp = pred.argmax(dim=1) 

        segmentation_classe = getTargetSegmentation2(labels)
        #print(segmentation_classe.size())
        #print(pred.size())
        dl=Dice_loss(pred, segmentation_classe)
        #print(" DiceLoss: {:.4f}, ".format(dl))
        dice.append(dl)
        #print(pp.size())
        for i in range(256):
            for j in range(256):
                if(pp.data[0,i,j]==1 and segmentation_classes.data[i,j]==1):
                    n=n+1
                elif(pp.data[0,i,j]==(2) and segmentation_classes.data[i,j]==2):
                    q=q+1
                elif(pp.data[0,i,j]==(3) and segmentation_classes.data[i,j]==3):
                    s=s+1
        correct += p.eq(segmentation_classes.view_as(p)).sum().item()      
      #labels=labels.squeeze(0)
      #print(metrics._calculate_multi_metrics(labels.squeeze(1),pred,4)[1])
    
    
    
    test_loss /= len(val_loader.dataset)
    dice = np.asarray(dice)
    dice = dice.mean()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset)*65536,
        100. * correct / len(val_loader.dataset)/65536))
    print("classe 1 : "+str(n)+"/38656")
    print("classe 2 : "+str(q)+"/49267")
    print("classe 3 : "+str(s)+"/66141")
    print(" DiceLoss: {:.4f}, ".format(dice))
