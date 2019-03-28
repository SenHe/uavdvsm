# Understanding-and-Visualizing-Deep-Visual-Saliency-Models-cvpr-2019
## Introduction
This is the demo of code, model and methods used in the paper.
There are some differences between the model used in the paper and this repository, the model used in the model is implemented in Tensorflow and the model in this repository is implemented in Pytorch(0.4.1).
## The Model Architecture:
![picture](archi.png)
The model is trained on Salicon database.  
Some saliency prediction examples on OSIE data  
![picture](sal_map.png)
## Synthetic data and annotation
### Data annotation
All the data annotation is done by myself using [labelme](https://github.com/wkentaro/labelme)
### Data link
[synthetic_data](https://drive.google.com/drive/folders/1wrdG1O5WgGl_ReoX5VGLKtroCuvzx2tv?usp=sharing)  
[OSIE-SR](https://drive.google.com/open?id=15iWBfNwktSq6KsNtAU1KRn0N3kVSOWHh)  
The segmentationclass folder contains the semantic level masks for each salient regions in the image,and the segmentationobject folder contains the instance level masks for each salient regions in the image.  
Some data examples:  
<img src="se1.jpg" width="280" height="200" /><img src="se2.jpg" width="280" height="200" /><img src="se3.jpg" width="280" height="200" />  
<img src="in1.jpg" width="280" height="200" /><img src="in2.jpg" width="280" height="200" /><img src="in3.jpg" width="280" height="200" />
