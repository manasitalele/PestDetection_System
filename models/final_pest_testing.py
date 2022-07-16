# -*- coding: utf-8 -*-
"""Final_Pest_testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DeacZmsYCV_T9x6UEoePzTnZDYF7WeTC
"""

from google.colab import drive
drive.mount('/content/drive')
import os
import numpy as np
import keras
import pandas as pd
from keras import models, layers, optimizers
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
image_size = 128
from PIL import Image, ImageTk

filename='/content/drive/My Drive/Corn_Dataset/Test/14/09274.jpg'
path_to_pest_img=filename
# Load an image in PIL format
pest_original = load_img(path_to_pest_img, target_size=(128, 128))

# Convert the PIL image to a numpy array
pest_numpy = img_to_array(pest_original)

# Convert the image into batch format
pest_batch = np.expand_dims(pest_numpy, axis=0)

# Prepare the image for the VGG model
pest_processed = vgg16.preprocess_input(pest_batch.copy())
pest_model=load_model('/content/drive/My Drive/pest_model_vgg.h5', compile=False)
# Get the predicted probabilities for each class
predictions = pest_model.predict(pest_processed)

lalbel2=predictions.max(1)
pred=np.where(predictions == lalbel2)
id=pred[1][0]

print(predictions)
print(pred[1])