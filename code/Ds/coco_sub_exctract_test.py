import pandas as pd

cat_2017 = r"C:\Users\rom21\fiftyone\coco-2017\test\labels.json"

#cat_2017 = r"E:\Downloads\coco2017\annotations\instances_train2017.json"
data_path = r"C:\Users\rom21\fiftyone\coco-2017\test\data"
#data_path = r"E:\Downloads\coco2017\train2017"
new_path = 'E:\COCO2017_Test'
import os
import shutil

import numpy as np
import sys, getopt
import json
classes = ["bicycle", "boat", "bus", "car",
           "cat", "dog", "motorcycle", "person"]

with open(cat_2017,'r') as COCO:
    js = json.loads(COCO.read())
catego_list = ['']*100
catego = js['categories']
annotations = js['annotations']
image_id = []
labels = []
for annotation in annotations:
    image_id.append(annotation['image_id'])
    labels.append(annotation['category_id'])

for cat in catego:
    catego_list[int(cat['id'])] = cat['name']

labels = np.array(labels)
catego_list = np.array(catego_list)
class_id = np.where(np.isin(catego_list, classes))[0]
image_id = np.array(image_id)
my_labels = labels[np.isin(labels, class_id)]
my_images = image_id[np.isin(labels, class_id)]
ds = {"bicycle":[], "boat":[], "bus":[], "car":[],
           "cat":[], "dog":[], "motorcycle":[], "person":[]}
for id in class_id:
    label_images = my_images[my_labels==id]
    not_label_images = my_images[~my_labels==id]

    for image in label_images:
        if np.unique(my_labels[my_images==image]).shape[0] == 1:
            while len(str(image)) != 12:
                image = '0'+str(image)
            image = os.path.join(data_path, image + '.jpg')
            ds[catego_list[id]].append(image)
if ~os.path.exists(new_path):
    os.mkdir(new_path)
for k in ds.keys():
    path_ds = os.path.join(new_path, k)
    if ~os.path.exists(path_ds):
        os.mkdir(path_ds)

for key in ds.keys():
    for val in ds[key]:
        path = os.path.join(new_path, key)
        shutil.copyfile(val, os.path.join(path, os.path.basename(val)))

#df = pd.DataFrame(ds)
pass
x=5