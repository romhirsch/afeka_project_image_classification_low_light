import xml.etree.ElementTree as ET
import os
import numpy as np
# Pass the path of the xml document
path = r"E:\Downloads\ILSVRC2012_bbox_val_v3"
name_image = []
class_name = []
for xml in os.listdir(path):
    full_path_file = os.path.join(path, xml)
    tree = ET.parse(full_path_file)
    root = tree.getroot()
    name_image.append(name_image)
    class_name.append(root[-1][0].text)
np.where('n02690373' == np.array(class_name))[0].shape
