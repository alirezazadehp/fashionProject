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

############################ROC################################
###############VGG-16
#vgg_conv = vgg16.VGG16(include_top=False, input_shape=(200, 114, 3))
#for layer in vgg_conv.layers:
#    layer.trainable = True
#    
#model = models.Sequential()
#model.add(vgg_conv)
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.2))
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

#vgg_conv = vgg16.VGG16(include_top=False, input_shape=(200, 114, 3))
###vgg_conv = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(200, 114, 3))
###
###
#for layer in vgg_conv.layers:
#    layer.trainable = True
###for layer in vgg_conv.layers[:-4]:    
#model = models.Sequential()
#model.add(vgg_conv)
#model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(6, activation='softmax'))
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
##############################################################################################Code    
base_model = resnet50.ResNet50
base_model = base_model(include_top=False, input_shape=(200, 114, 3))
for layer in base_model.layers:
    layer.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
#
#
#model.compile(loss='binary_crossentropy',
#              optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
#              metrics=['acc'])   
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) 
   
# 
    
# checkpoint
#filepath="weights.best.vgg16.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]   
#
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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 114),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(200, 114),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])    

history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=15,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
#      callbacks=callbacks_list,
      verbose=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
###########################PLOT ROC ############################
#true_classes = validation_generator.classes
#true_classes = np_utils.to_categorical(true_classes, 4)
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
#for i in range(4):
#    fpr[i], tpr[i], _ = roc_curve(true_classes[:, i], predictions[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
    
##########################ROC Curve########################################
#true_classes = validation_generator.classes
#predictions = model.predict_generator(generator = validation_generator, steps = validation_generator.samples/validation_generator.batch_size)
#true_classes = np_utils.to_categorical(true_classes, 6)
#
#lw = 2
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(6):
#    fpr[i], tpr[i], _ = roc_curve(true_classes[:, i], predictions[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(true_classes.ravel(), predictions.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#
#
#
#
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
#
#
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(6):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
#
#mean_tpr /= 6
#
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
#
#plt.figure()
#
#lablels = ['casual', 'girly', 'others', 'sport', 'tomboy', 'vacation']
#
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'deeppink', 'green'])
#for i, color in zip(range(6), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='{0}'
#             ''.format(lablels[i]))
#
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic for the VGG-16')
#plt.legend(loc="lower right")
#plt.show()