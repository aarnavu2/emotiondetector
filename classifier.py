import tensorflow as tf
import os

import cv2 #computer vision library
import imghdr #allows us to check file extensions

from matplotlib import pyplot as plt
print(os.listdir('abc'))

gpus = tf.config.experimental.list_physical_devices('GPU') #gives all gpus available
for gpu in gpus:                                           #loops through all gpus and tells tensorflow minimizes takes memory
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data' #just stores the name of our data directory
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

""" this part just for understanding cv2 and matplot lib

img = cv2.imread(os.path.join('data', 'happy', '_happy_jumping_on_beach-40815.jpg')) #stores random image from happy directory in numpy array
print(img.shape) #gives dimensions of numpy array with depth 3 for rgb
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #use maplotlib to visualize includes next line
plt.show() 
"""
#This part actually does stuff
for image_list in os.listdir(data_dir): #loops through data folder which will go to happy folder then sad folder
    for image in os.listdir(os.path.join(data_dir, image_list)): #goes through each image name
        image_path = os.path.join(data_dir, image_list, image) #gives each image their image path
        try:
            img = cv2.imread(image_path) #make sure it loads in cv2
            tip = imghdr.what(image_path) #gives us extension
            if tip not in image_exts: #if our image is of a weird extension
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path) #deletes image

        except Exception as e:
            print("issue with image {}".format(image_path))


happy_images = os.listdir(os.path.join(data_dir, 'happy')) #list of all names of happy images
sad_images = os.listdir(os.path.join(data_dir, 'sad')) #list of all names of sad images


