#03/04/2018
## tomato pictures grown/unripe classification


import os
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sklearn.metrics as sklm
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense,  Dropout, Activation
from keras.layers import Sequential, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from scipy import misc

img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)


#####disable CUDA 
disable_CUDA=True


#####if disable CUDA
if disable_CUDA :
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1 !!!!
#####disable CUDA

###########################  GPU vram optimization
if not disable_CUDA :
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  K.set_session(sess)

###########################


    
def pretrain_classifier_top ():
    model_vgg16 = VGG16(weights='imagenet', include_top=False)  ### no top layers , imagenet weights
    
    #freeze all layers
    for layer in model_vgg16.layers:
      layer.trainable = False
      
    #Set input format
    keras_input = Input(shape=(224,224,3), name = 'image_input')
    
    #new vgg16 model 
    output_vgg16_conv = model_vgg16(keras_input)
    
    #New classifier layers
    new = Flatten(name='flatten')(output_vgg16_conv)
    new = Dense(4096, activation='relu', name='fc1')(new)
    new = Dense(4096, activation='relu', name='fc2')(new)
    new = Dense(2, activation='softmax', name='predictions')(new)
    print("compiling")
    pretrained_model = Model(inputs=keras_input, outputs=new)
    #pretrained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    pretrained_model.compile(loss='categorical_crossentropy',  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    print("model compiled")
    
    return pretrained_model  

def img_gen_load(): #### images auto generator for training 
  
  train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
  
  train_generator = train_datagen.flow_from_directory(
        'train',  # this is the train directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=10,
        class_mode='categorical')    
  return train_generator    

def validation_generator(): #### images auto generator for validation 
  
  validation_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
  
  validation_generator = validation_datagen.flow_from_directory(
        'validation',  # this is the validation directory
        target_size=(224, 224),  # all images will be resized to 224x224
        batch_size=36,
        shuffle=False,
        class_mode='categorical')    
  return validation_generator 

  
def prep_valid(input):  ####prepr export to csv
  out1=[]
  out2=[]
  for l in input:
    out1.append(l[0])
    out2.append(l[1])
  return np.asarray(out1),np.asarray(out2)  
  

  
print("train generator")  
train_generator = img_gen_load()  
print("model")  
model2 = pretrain_classifier_top() 
print("Starting training")
model2.fit_generator(
        train_generator,
        steps_per_epoch=111, # batch_size : 372 images * 3  / 10 
        epochs=40)
print("Saving model")          
model2.save("model2_117_40_01")        
print("validation generator")        
validation_generator=validation_generator()
validation_generator.class_indices  ###classes
y=validation_generator.classes      ###ylabels
print("Model predictions")
valid=model2.predict_generator(validation_generator,steps=1,verbose=1)        
        
score = model2.evaluate_generator(validation_generator,steps=1)
print("Loss - accuracy")

print("--")
print("Outputting predictions in two np array")
grow_pred, unripe_pred = prep_valid(valid)
print("Use unripe_pred vs y for matrix confusion, roc, recall, precision, etc...")
   
