import csv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import keras
from sklearn import manifold
from itertools import groupby
import random
from sklearn.cluster import KMeans
import math
from keras.applications import vgg16, inception_resnet_v2, resnet50, vgg19
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

#from keras.applications.vgg16 import preprocess_input
#from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D, Dense,Input, Flatten
from keras.applications.vgg19 import preprocess_input
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from metrics import *
from keras import regularizers
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda
from keras.layers import Dropout

weight_decay = 1e-6
############################ROC################################   weight_decay = 1e-6
###############VGG-16
#vgg_conv = vgg16.VGG16(include_top=False, input_shape=(200, 114, 3))
#for layer in vgg_conv.layers:
#    layer.trainable = True
#    
#model = models.Sequential()
#model.add(vgg_conv)
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(6, activation='softmax'))
#model.load_weights("weights.best.vgg16.hdf5")
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
###############VGG-19

#vgg_conv1 = vgg19.VGG19(include_top=False, input_shape=(200, 114, 3))
#for layer in vgg_conv1.layers:
#    layer.trainable = True
#    
#model1 = models.Sequential()
#model1.add(vgg_conv1)
#model1.add(layers.Flatten())
#model1.add(layers.Dense(1024, activation='relu'))
#model1.add(layers.Dropout(0.5))
#model1.add(layers.Dense(6, activation='softmax'))
#model1.load_weights("weights.best.vgg19.hdf5")
#model1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

###############Resnet50

#base_model = resnet50.ResNet50
#base_model = base_model(include_top=False, input_shape=(200, 114, 3))
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#predictions = Dense(6, activation='softmax')(x)
#model2 = Model(inputs=base_model.input, outputs=predictions)
#for layer in base_model.layers:
#    layer.trainable = True
#    
#model2.load_weights("weights.best.resnet50.hdf5")
#model2.compile(loss='binary_crossentropy',
#              optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
#              metrics=['acc'])



##################################################################
#a = input_shape=(200, 114, 3)
#y = Input(shape=(4,))



#vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False,input_shape=inputT)
###vgg_conv = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(200, 114, 3))
###
###
#
#
#for layer in vgg_conv.layers:
#    layer.trainable = True
####for layer in vgg_conv.layers[:-4]:    
#model = models.Sequential()
#model.add(vgg_conv)
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#last = model.output


# Block 1
#x = layers.Conv2D(64, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block1_conv1')(input)
#x = layers.Conv2D(64, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block1_conv2')(x)
#x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
#
## Block 2
#x = layers.Conv2D(128, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block2_conv1')(x)
#x = layers.Conv2D(128, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block2_conv2')(x)
#x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
#
## Block 3
#x = layers.Conv2D(256, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block3_conv1')(x)
#x = layers.Conv2D(256, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block3_conv2')(x)
#x = layers.Conv2D(256, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block3_conv3')(x)
#x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
#
## Block 4
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block4_conv1')(x)
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block4_conv2')(x)
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block4_conv3')(x)
#x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
#
## Block 5
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block5_conv1')(x)
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block5_conv2')(x)
#x = layers.Conv2D(512, (3, 3),
#                  activation='relu',
#                  padding='same',
#                  name='block5_conv3')(x)
#x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

input = Input(shape=(200, 114, 3))
label = Input(shape=(6,))


#x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input)
#x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
#x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
#x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
#x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)




model_vgg16_conv = vgg16.VGG16(weights='imagenet', include_top=False)
output_vgg16_conv = model_vgg16_conv(input)




#vgg_conv1 = vgg16.VGG16(weights='imagenet',include_top=False, input_shape=input)
#for layer in vgg_conv1.layers:
#    layer.trainable = True
# 




#x = Flatten(name='flatten')(output_vgg16_conv)

#x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
#x = MaxPooling2D(pool_size=(2, 2))(x)


x = BatchNormalization()(output_vgg16_conv)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(512, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
output = ArcFace(6, regularizer=regularizers.l2(weight_decay))([x, label])
#output1 = CosFace
#output = [output1,output2]
model = Model([input, label], output)

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


#'Adam'

#model1 = Model([a, y], output)
##model.add(layers.Dense(4, activation='softmax'))
#model1.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
#base_model = resnet50.ResNet50
#base_model = base_model(weights='imagenet', include_top=False, input_shape=(200, 114, 3))
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#predictions = Dense(6, activation='softmax')(x)
#model = Model(inputs=base_model.input, outputs=predictions)
#for layer in base_model.layers:
#    layer.trainable = True
#
#model.compile(loss='binary_crossentropy',
#              optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
#              metrics=['acc'])   
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) 
   
# 
    
# checkpoint
filepath="weights.best.vgg16ArcFace3.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]   

train_dir = "Train"
test_dir = "Test"



BATCH_SIZE = 100
val_batchsize = 10

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=True,
    fill_mode='nearest')

def generate_generator_multiple(generator,dir1):
    genX1  = generator.flow_from_directory(
        dir1,
        target_size=(200, 114),
        batch_size=100,
        class_mode='categorical')
    while True:
        X1i = genX1.next()
        yield [X1i[0], X1i[1]], X1i[1]
    

train_generator=generate_generator_multiple(train_datagen,
                                   train_dir)
                                          

validation_generator=generate_generator_multiple(validation_datagen,
                                  test_dir)
                                   






#model.compile(loss='categorical_crossentropy',
#              optimizer=optimizers.RMSprop(lr=1e-4),
#              metrics=['acc'])    

history = model.fit_generator(
      train_generator,
      steps_per_epoch=2757,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=6782,
      callbacks=callbacks_list,
      verbose=1)
###########################PLOT ROC ############################
#true_classes = validation_generator.classes
#true_classes = np_utils.to_categorical(true_classes, 6)
#
#predictions = model.predict_generator(generator = validation_generator, steps = validation_generator.samples/validation_generator.batch_size)
#predictions1 = model1.predict_generator(generator = validation_generator, steps = validation_generator.samples/validation_generator.batch_size)          
#predictions2 = model2.predict_generator(generator = validation_generator, steps = validation_generator.samples/validation_generator.batch_size)  
############################################################          
#model13 = model.fit_generator(generator(a), samples_per_epoch=ll, nb_epoch=23)

#model13 = model.fit_generatorfit(X_trainTotal, trainImgClassTotal, validation_data=(X_testTotal, testImgClassTotal), batch_size=100, verbose=1, epochs=23)

##########tpr#######
#lw = 2
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(6):
#    fpr[i], tpr[i], _ = roc_curve(true_classes[:, i], predictions[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
    
