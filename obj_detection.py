import os
import plotly.express as px
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

# resnet50
from keras.applications import ResNet50

HYP = dict(
    batch_size = 16,
    img_size = (128,128),
    epochs = 25
)


train_dir ='C:/Users/vedak/Downloads/archive (4)/Fast Food Classification V2/Train'
valid_dir = 'C:/Users/vedak/Downloads/archive (4)/Fast Food Classification V2/Valid'
test_dir = 'C:/Users/vedak/Downloads/archive (4)/Fast Food Classification V2/Test'

class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)

# Print
print("No. Classes : {}".format(num_classes))
print("Classes     : {}".format(class_names))

# Augment train set only
train_data_generator = ImageDataGenerator(
                    rotation_range=15,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest'
)

val_data_generator = ImageDataGenerator()

test_data_generator = ImageDataGenerator()



train_generator = train_data_generator.flow_from_directory(
    train_dir, target_size=HYP['img_size'],
    class_mode='categorical', 
    batch_size=HYP['batch_size']
)

validation_generator = val_data_generator.flow_from_directory(
    valid_dir, target_size=HYP['img_size'],
    class_mode='categorical',
    batch_size=HYP['batch_size']
)

test_generator = test_data_generator.flow_from_directory(
    test_dir, target_size=HYP['img_size'],
    class_mode='categorical',
    batch_size=HYP['batch_size']
)


from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * tf.math.exp(0.001 * (5 - epoch))
    
lr_scheduler = LearningRateScheduler(scheduler)
callbacks = [lr_scheduler]


model = Sequential()

model.add(
    ResNet50(
        include_top=False,
        pooling='avg',
        weights='imagenet'
    )
)

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.summary()

model.compile(optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy'])


history = model.fit(
    train_generator,
    epochs=HYP['epochs'],
    validation_data=validation_generator,
    callbacks=callbacks
)

model.save('FastfoodModel.h5')