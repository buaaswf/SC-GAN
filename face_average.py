import os
import cv2
import numpy as np

def _mkdirs(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

#path0='./cluster_data/closed/0'
#path1='./cluster_data/closed/1'
#path2='./cluster_data/closed/2'
path = 'pca_cluster_data/smile/crop/2_nosmile'
#output = np.zeros((218,178,3), np.float32());
output = np.zeros((128,128,3), np.float32());
numImages=0;
for name in os.listdir(path):
    img = cv2.imread(os.path.join(path, name));
    # Convert to floating point
    img = np.float32(img) / 255.0;
    output=output+img;
    #if numImages > 11545 and numImages <= 11546:
    if numImages % 100 == 0:
        print(numImages)
        #print(name)
        #print(img)
        #print(output.mean())
    numImages=numImages+1;
output = output / numImages;
print(output)
#cv2.imshow('image', output);
_mkdirs('./cluster_data/close/')
_mkdirs('./cluster_data/nosmile/')
cv2.imwrite('./cluster_data/nosmile/2.jpg',output*255)
#cv2.waitKey(0);
