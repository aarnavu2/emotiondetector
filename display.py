import tensorflow as tf
import os

import cv2 #computer vision library
import imghdr #allows us to check file extensions

from matplotlib import pyplot as plt

import numpy as np

model = tf.keras.models.load_model("happysadmodel.h5")

data = tf.keras.utils.image_dataset_from_directory('data') #builds data pipeline
#this takes a bunch of image data and pretty much seperates it into batches

data = data.map(lambda x,y: (x/255, y)) #let's us do a transformation using a lambda function

train_size = int(len(data)*.7)
val_size = int(len(data)*.2) + 1
test_size = int(len(data)*.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

### Evaluate

"""pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X,y = batch
    yhat = model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat),
    acc.update_state(y,yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy}')"""

### Testing

img_loc = input("please type file location of your test image\n")
img = cv2.imread(img_loc)
resize = tf.image.resize(img, (256,256))
output = model.predict(np.expand_dims(resize/255,0))

if (output > 0.5):
    print("image is sad")
else:
    print("image is happy")


"""img_happy = cv2.imread('happy_test.jpg')
img_sad = cv2.imread('sad_test.jpg')

resize_happy = tf.image.resize(img_happy, (256,256))
resize_sad = tf.image.resize(img_sad, (256,256))


#expect a batch

test_happy = model.predict(np.expand_dims(resize_happy/255,0))

""if (test_happy > 0.5):
    print("image is sad")
else:
    print("image is happy")

#sad image


test_sad = model.predict(np.expand_dims(resize_sad/255,0))

if (test_sad > 0.5):
    print("image is sad")
else:
    print("image is happy")"""

### Save the model

#test_load = tf.keras.models.load_model('happysadmodel.h5')
#test_pred = test_load.predict()
