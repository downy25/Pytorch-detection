import os
import numpy as np
import utils
import transforms as T
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import cv2
import sys


class RobotSizeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPGImages"))))
        self.annot = list(sorted(os.listdir(os.path.join(root, "Annotation"))))
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "JPGImages", self.imgs[idx])
        annot_path = os.path.join(self.root, "Annotation", self.annot[idx])
        img = Image.open(img_path).convert("RGB")

        # xml 파일 파싱
        tree=ET.parse(annot_path)
        root = tree.getroot()

        #bounding box save
        bounding=[]
        for bbox in root.iter('bndbox'):
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bounding.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(bounding, dtype=torch.float32)
 
        # area of each bounding box
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # small robot(label)=0, big robot(label)=1
        labels=[]
        for label in root.iter('object'):
            if label.find('name').text == "big robot":
                labels.append(0)  
            elif label.find('name').text == "small robot":
                labels.append(1)  
            
        labels = torch.as_tensor(labels)
 
        # define id for this image
        image_id = torch.tensor([idx])
 
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
 
        # put it into the dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img,target= self.transforms(img,target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)















# cv2라이브러리 활용하여 바운딩박스정보 출력
# img = cv2.imread('IMG_7597.JPG')
# if img is None:
#     print('Image load failed')
#     sys.exit()

# box=label_batch['boxes']
# labels=[]
# for label in label_batch['labels']:
#     if label==0:
#         labels.append("small robot")
#     elif label==1:
#         labels.append("big robot")

# bounding=[]
# for j in range(2):
#     for point in box[j]:
#         bounding.append(point)
# pt1_x=int(bounding[0]) 
# pt1_y=int(bounding[1])
# pt2_x=int(bounding[2])
# pt2_y=int(bounding[3])
# pt3_x=int(bounding[4])
# pt3_y=int(bounding[5])
# pt4_x=int(bounding[6])
# pt4_y=int(bounding[7])
# color=(0,255,255)
# font =  cv2.FONT_HERSHEY_PLAIN
# color_font=(0,0,255)
# pt1=(pt1_x,pt1_y)
# pt2=(pt2_x,pt2_y)
# pt3=(pt3_x,pt3_y)
# pt4=(pt4_x,pt4_y)    
# cv2.rectangle(img,pt1,pt2,color,thickness=4)
# cv2.rectangle(img,pt3,pt4,color,thickness=4)
# img = cv2.putText(img, labels[0], (pt1_x, pt1_y), font, 8, color_font, 6, cv2.LINE_AA)
# img = cv2.putText(img, labels[1], (pt3_x, pt3_y), font, 8, color_font, 6, cv2.LINE_AA)
# resized_img_1 = cv2.resize(img, dsize=(880,495), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("img",resized_img_1) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt라이브러리 사용해서 dataloader 찍어보기        
# fig = plt.figure(figsize=(15, 6))
# num_epochs = 1
# for j in range(num_epochs):
#     img_batch, label_batch = next(iter(data_loader))
#     img = img_batch[0]
#     ax = fig.add_subplot(1, 1, j + 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_title(f'Epoch {j}:', size=15)
#     ax.imshow(img.permute(1, 2, 0))
# #plt.savefig('figures/14_16.png', dpi=300)
# plt.show()