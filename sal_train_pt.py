import torch as t
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import os
import glob
from PIL import Image
import numpy as np
from scipy.misc import imread
from logger import Logger
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


class salnet(nn.Module):
    def __init__(self):
        super(salnet,self).__init__()
        
        vgg16 = models.vgg16()
        
        encoder = list(vgg16.features.children())[:-1]
        self.encoder = nn.Sequential(*encoder)
        #self.fine_tuning()
        #self.compress = nn.Conv2d(512,128,1,padding=0,bias=False)
        self.decoder = nn.Conv2d(512,1,1,padding=0,bias=False)
    def forward(self,im):
        e_out = self.encoder(im)
        #fea = self.compress(e_out)
        x = self.decoder(e_out)
        x = nn.functional.interpolate(x,size=(480,640),mode='bilinear',align_corners=False)
        x = x.squeeze(1)
        mi = t.min(x.view(-1,480*640),1)[0].view(-1,1,1)
        ma = t.max(x.view(-1,480*640),1)[0].view(-1,1,1)
        n_x = (x-mi)/(ma-mi)
        return n_x

def NSS(sal,fix):
    m = t.mean(sal.view(-1,480*640),1).view(-1,1,1)
    std = t.std(sal.view(-1,480*640),1).view(-1,1,1)
    n_sal = (sal-m)/std
    s_fix = t.sum(fix.view(-1,480*640),1)
    ns = n_sal*fix
    s_ns = t.sum(ns.view(-1,480*640),1)
    nss = t.mean(s_ns/s_fix)
    return -nss

def cor(fea):
    s = fea.shape
    fea1 = fea.view(s[0],s[1],14*14)
    fea2 = fea1.permute(0,2,1)
    rela = t.bmm(fea1,fea2)
    return t.sum(rela)/(10*128*128)

transform = T.Compose([
        #T.Resize(224),
        #T.CenterCrop(224),
	T.ToTensor(),
	T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

class im_la(data.Dataset):
    def __init__(self,root1,root2,transforms=None):
        imgs = glob.glob(os.path.join(root1,'*.jpg'))
        self.imgs = [img for img in imgs]
        self.labels = [root2+img.split('/')[-1][0:-4]+'.png' for img in imgs]
        self.transforms = transforms
    def __getitem__(self,index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).resize((224,224))
        if self.transforms:
            data = self.transforms(pil_img)
        label_path = self.labels[index]
        label =t.from_numpy(imread(label_path)/255.0)
        label = label.float()
        return data,label
    def __len__(self):
        return len(self.imgs)

tr_dataset = im_la('/media/Data/Sen/Desktop/sal_vis/train_im/','/media/Data/Sen/Desktop/sal_vis/train_fix/',transforms=transform)
va_dataset = im_la('/media/Data/Sen/Desktop/sal_vis/val_im/','/media/Data/Sen/Desktop/sal_vis/val_fix/',transforms = transform)

tr_dataloader = DataLoader(tr_dataset,batch_size=20,shuffle=True,num_workers=0,drop_last=True)
va_dataloader = DataLoader(va_dataset,batch_size=10,shuffle=False,num_workers=0,drop_last=True)

model = salnet()

model = model.to(device)
print(model)
############################
###optimizer######
for param in model.encoder.parameters():
    print(param.requires_grad)
for param in model.decoder.parameters():
    print(param.requires_grad)

optimizer = t.optim.Adam(model.parameters(),lr=0.005)
iter_per_epoch = len(tr_dataloader)
va_iter_per_epoch = len(va_dataloader)
print(iter_per_epoch)
print(va_iter_per_epoch)
total_step = 20*500
##########
logger = Logger('./vgg_s_logs')
###############

for step in range(total_step):
    #reset data loader
    if (step)%iter_per_epoch==0:
        tr_data_iter = iter(tr_dataloader)
    #fetch image and labels
    images,labels = next(tr_data_iter)
    images,labels = images.to(device), labels.to(device)
    #forward pass
    outputs = model(images)
    #print(fea_out.shape)
    #print(outputs.shape)
    loss = NSS(outputs,labels)
    #print(loss1)
    #loss2 = cor(fea_out)
    #print(loss2)
    #loss = 10*loss1+loss2/1000000
    #print(loss)
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(loss.item())
    if (step+1)%50==0:
        info = {'-nss':loss.item()}
        #loss
        for tag,value in info.items():
            logger.scalar_summary(tag,value,step+1)
        #weight
        #for tag,value in vgg16.named_parameters():
        #    tag = tag.replace('.','/')
        #    logger.histo_summary(tag,value.data.cpu().numpy(),step+1)
        #    logger.histo_summary(tag+'/grad',value.grad.data.cpu().numpy(),step+1)
        #image
        #info = {'images':images.view(-1,28,28)[:10].cpu().numpy()}
        #for tag,images in info.items():
        #    logger.image_summary(tag,images,step+1)
    if (step+1)%500==0:
        for param_group in optimizer.param_groups:
            param_group['lr']*=0.8
    if (step+1)%500==0:
        t.save(model.state_dict(),'./vgg_s_model/'+str(step+1)+'.pt')
    if (step+1)%500==0:
        model.eval()
        count1 = t.zeros(1)
        #count2 = t.zeros(1)
        va_data_iter = iter(va_dataloader)
        for j in range(va_iter_per_epoch):
            images,labels = next(va_data_iter)
            images,labels = images.to(device), labels.to(device)
        #forward pass
            outputs = model(images)
            loss = -NSS(outputs,labels)
            #loss2 = cor(fea_out)/1000
            count1+=loss.item()
            #count2+=loss2.item()
        va_acc = count1/(j+1)
        #va_cor = count2/(j+1)
        va_info = {'va_nss':va_acc}
        #loss
        for tag,value in va_info.items():
            logger.scalar_summary(tag,value,step+1)
        model.train()
