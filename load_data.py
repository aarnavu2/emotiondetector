
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2

data = tf.keras.utils.image_dataset_from_directory('data') #builds data pipeline
#this takes a bunch of image data and pretty much seperates it into batches

data = data.map(lambda x,y: (x/255, y)) #let's us do a transformation using a lambda function

scaled_iterator = data.as_numpy_iterator() #Allow access to data pipeline
#this lets us go batch tobatch

batch = scaled_iterator.next() #accesses data

#Images represented as numpy arrays
# Class 1 = sad people
# Class 2 = happy people


#### Split data #####


train_size = int(len(data)*.7)
val_size = int(len(data)*.2) + 1
test_size = int(len(data)*.1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#### Training

tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), 1, activation = 'relu', input_shape=(256,256,3)), #input layer, has 16 filters that are 3 by 3 and move by 1 pixel, relu activation means any output that was below 0 becomes 0, positive vals are unchanged, good for nonlinear
    tf.keras.layers.MaxPooling2D(), #reduces spacial dimensions of previous layers

    tf.keras.layers.Conv2D(32, (3,3), 1, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, (3,3), 1, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(), #we need a 1d input for dense layers thats what flatten does

    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid') #converts to a range of 0-1 for happy and sad

])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["accuracy"])


### train

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir) #for logging

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

print(hist)

fig = plt.figure()
plt.plot(hist.history["loss"], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('loss', fontsize=20)
plt.legend(loc="upper left")

fig = plt.figure()
plt.plot(hist.history["accuracy"], color = 'teal', label = 'loss')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'val_loss')
fig.suptitle('accuracy', fontsize=20)
plt.legend(loc="upper left")


### Evaluate

pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X,y = batch
    yhat = model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat),
    acc.update_state(y,yhat)

print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy: {acc.result().numpy}')

### Testing

img = cv2.imread('happy_test.jpg')
resize = tf.image.resize(img, (256,256))

#expect a batch

test_happy = model.predict(np.expand_dims(resize/255,0))
print(test_happy)

#sad image

img2 = cv2.imread('sad_test.jpg')
resize2 = tf.image.resize(img, (256,256))


test_sad = model.predict(np.expand_dims(resize2/255,0))
print(test_sad)

### Save the model

model.save('happysadmodel.h5')
#test_load = tf.keras.models.load_model('happysadmodel.h5')
#test_pred = test_load.predict()

plt.show()