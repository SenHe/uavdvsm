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
from scipy.misc import imread,imsave
from logger import Logger
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


class salnet(nn.Module):
    def __init__(self):
        super(salnet,self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        
        encoder = list(vgg16.features.children())[:-1]
        self.encoder = nn.Sequential(*encoder)
        self.fine_tuning()
        self.decoder = nn.Conv2d(512,1,1,padding=0,bias=False)
    def forward(self,x):
        e_x = self.encoder(x)
        d_x = self.decoder(e_x)
        d_x = nn.functional.interpolate(d_x,size=(480,640),mode='bilinear',align_corners=False)
        d_x = d_x.squeeze(1)
        mi = t.min(d_x.view(-1,480*640),1)[0].view(-1,1,1)
        ma = t.max(d_x.view(-1,480*640),1)[0].view(-1,1,1)
        n_x = (d_x-mi)/(ma-mi)
        return e_x,n_x
    def fine_tuning(self):
        i=0
        for param in self.encoder.parameters():
            if i<20:
                param.requires_grad=False
                i+=1

def NSS(sal,fix):
    m = t.mean(sal.view(-1,480*640),1).view(-1,1,1)
    std = t.std(sal.view(-1,480*640),1).view(-1,1,1)
    n_sal = (sal-m)/std
    s_fix = t.sum(fix.view(-1,480*640),1)
    ns = n_sal*fix
    s_ns = t.sum(ns.view(-1,480*640),1)
    nss = t.mean(s_ns/s_fix)
    return -nss

transform = T.Compose([
        #T.Resize(224),
        #T.CenterCrop(224),
	T.ToTensor(),
	T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

class im_la(data.Dataset):
    def __init__(self,root1,transforms=None):
        imgs = glob.glob(os.path.join(root1,'*.jpg'))
        self.imgs = [img for img in imgs]
        self.transforms = transforms
    def __getitem__(self,index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path).resize((448,448))
        if self.transforms:
            data = self.transforms(pil_img)
        return data,img_path
    def __len__(self):
        return len(self.imgs)
va_dataset = im_la('/home/sen/Desktop/fingra/dog/train/',transforms = transform)
va_dataloader = DataLoader(va_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=True)

model = salnet()
model.load_state_dict(t.load('./model/4000.pt'))
model = model.to(device)
model.eval()
print(model)
############################
###optimizer######
va_iter_per_epoch = len(va_dataloader)
print(va_iter_per_epoch)
va_data_iter = iter(va_dataloader)
files = glob.glob(os.path.join('/media/Data/Sen/Desktop/mit_1003_im','*.jpeg'))
N = len(files)
print(N)
for i in range(N):
    print(i)
    im = Image.open(files[i]).resize((448,448))
    t_im = transform(im).unsqueeze(0)
    t_im = t_im.to(device)
    name = files[i].split('/')[-1][0:-4]
    eo,pred = model(t_im)#.squeeze(0)
    pred = pred.squeeze(0)
    eo = eo.squeeze(0)
    n_eo = eo.cpu().detach().numpy()
    n_pred = pred.cpu().detach().numpy()
    imsave('/media/Data/Sen/Desktop/mit_free_sal/'+name+'.jpg',n_pred)
    #np.save('./exa/fea/'+name+'.npy',n_eo)
