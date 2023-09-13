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


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def runTrainingUnlabel():
    #torch.set_grad_enabled(True)
    print('-' * 40)
    print('~~~~~~  Starting the training... ~~~~')
    print('-' * 40)

    lr =  0.0001   # Learning Rate
    epoch = 1 # Number of epochs
    batch_size = 15
    
    net = mdf.MedFormer(1, 4)
    checkpoint = torch.load('<saved_model_path>')
    net.load_state_dict(checkpoint)
    
    ema_net = net

    
    root_dir = './data/'

    print(' Dataset: {} '.format(root_dir))
    
     ## DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set_full = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)
    
    
    utrain_set_full = medicalDataLoader.MedicalImageDataset('unlabeled',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    utrain_loader_full = DataLoader(train_set_full,
                              batch_size=batch_size,
                              worker_init_fn=np.random.seed(0),
                              num_workers=0,
                              shuffle=True)
    
    
    

    num_classes = 4 # NUMBER OF CLASSES

    print("~~~~~~~~~ Creating the MedFormer model ~~~~~~~~")
    modelName = 'Test_Model'
    print(" Model Name: {}".format(modelName))

    print("Total params: {0:,}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    Dice_loss = DiceLoss(classes=[1,2,3], ignore_index=0)
    CE_loss = nn.CrossEntropyLoss()
    
    directory = 'Results/Statistics/' + modelName

    print("~~~~~~~~~ Starting the training ~~~~~~~~")
    if os.path.exists(directory)==False:
        os.makedirs(directory)

    ## START THE TRAINING
    n=0
    ## FOR EACH EPOCH
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()
    ema_net.train()
    n_iter=0
    for i in range(epoch):
        
        lossEpoch = []
        DSCEpoch = []
        DSCEpoch_w = []
        num_batches = len(train_loader_full)
        
        ## FOR EACH BATCH
        if(n<100000):
          for (j, data),(k,unsupdata) in zip(enumerate (train_loader_full), enumerate (utrain_loader_full)):
              n=n+1

              images, labels, img_names = data[1], data[2], data[3]
              imagesUnLab = unsupdata[1]
              ### From numpy to torch variables
              labels = to_var(labels)            
              images = to_var(images)             
              imagesUnLab = to_var(imagesUnLab)
              noise = torch.clamp(torch.randn_like(imagesUnLab) * 0.1, -0.2, 0.2)
              ema_inputs = imagesUnLab + noise
              labels_long = labels.type(torch.LongTensor)
              segmentation_classes = getTargetSegmentation(labels)

              pred = net(images)
              m = nn.Softmax(dim=1)
              
              
              outputs = net(imagesUnLab) 
              outputs_soft=m(outputs)
                

              ema_outputs = ema_net(imagesUnLab) 
              ema_outputs=Variable(ema_outputs.detach().data, requires_grad=False)
              ema_outputs_soft=m(ema_outputs)
              consistency_loss = torch.mean((outputs-ema_outputs)**2)
              pred=net(images)
              pred_soft = m(pred)
              dl=Dice_loss(pred_soft, segmentation_classes)
              cl=CE_loss(pred, segmentation_classes)
              
              
              loss = (1/2)*((3/4)*Dice_loss(pred_soft, segmentation_classes)+(1/4)*CE_loss(pred, segmentation_classes))+(1/2)*consistency_loss
              lossTotal = loss 
              optimizer.zero_grad()
              for param in net.parameters():
                param.requires_grad = True
              loss.backward()
              optimizer.step()
              
              lossEpoch.append(lossTotal.cpu().data.numpy())
              n_iter=n_iter+1
              printProgressBar(j + 1, num_batches,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f}, ".format(lossTotal)+" DiceLoss: {:.4f}, ".format(dl)+" CE: {:.4f}, ".format(cl)) 
        alpha = min(1 - 1 / (n_iter + 1), 0.99)
        update_ema_variables(net, ema_net, 0.999, n_iter)
        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()
        
        
        if not os.path.exists('./models/' + modelName):
            os.makedirs('./models/' + modelName)

        torch.save(net.state_dict(), './models/' + modelName + '/' + 'UnSupMy_Model3' + '_Epoch')
        
        
runTrainingUnlabel()