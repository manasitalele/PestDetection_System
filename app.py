
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import h5py
import numpy as np
import keras
import pandas as pd
import tensorflow as tf
import cv2

#Keras Libraries
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.preprocessing import image
from keras import models, layers, optimizers
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import ImageDataGenerator
#from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from PIL import ImageTk,Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

image_size = 128
imglist=[]
default_image_size = tuple((256, 256))
#Define the flask app
app = Flask(__name__)

#-----------------------Pest Image Processing Function--------------------------
def model_predict_pest(img_path):
    # Load an image in PIL format
    pest_original = load_img(img_path, target_size=(image_size, image_size))


    # Convert the PIL image to a numpy array
    pest_numpy = img_to_array(pest_original)


    # Convert the image into batch format
    pest_batch = np.expand_dims(pest_numpy, axis=0)


    # Prepare the image for the VGG model
    pest_processed = vgg16.preprocess_input(pest_batch.copy())
    pest_model=load_model('models/pest_final_model.h5', compile=False)
    print("model loaded!!!")
    # Get the predicted probabilities for each class
    predictions = pest_model.predict(pest_processed)
    print("Predictions:",predictions)
    return predictions

#---------------------------------------Disease Image Processing Function-----------------------------------
def model_predict_disease(image_path_disease):
    # Load an image in PIL format
    model= load_model('models/model.h5',compile=False)
    print("DISEASE MODEL LOADED")
    image = cv2.imread(image_path_disease)
    print("IMage READ........................HEYYYY!!!",image)
    image = cv2.resize(image, default_image_size)   
    print("Resize Done!!!!")
    img= img_to_array(image)
    print("Convert to ARRAY!!!!!")
    imglist.append(img)
    print("APPENDED..........!!!!")
    np_image_list = np.array(imglist, dtype=np.float16) / 225.0
    print("NP_IMAGE_LIST:",np_image_list)
    prediction=model.predict(np_image_list)
    print("PREDICTIONS_DISEASE:",prediction)
    return prediction


#----------------------------Main page--------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    # Main page
    print("Page Started!!!!:)")
    return render_template('index.html')


@app.route('/pest', methods=['GET'])
def pest():
    # Main page
    print("pest Page Started!!!!:)")
    return render_template('pestupload.html')


@app.route('/disease', methods=['GET'])
def disease():
    # Main page
    print("disease Page Started!!!!:)")
    return render_template('diseaseupload.html')

@app.route('/pestmanage', methods=['GET'])
def pest_manage():
    # Main page
    print("Pest Management Page Started!!!!:)")
    return render_template('pestmanage.html')

@app.route('/diseasemanage', methods=['GET'])
def disease_manage():
    # Main page
    print("Disease Management Page Started!!!!:)")
    return render_template('diseasemanage.html')



#----------------------Prediction route for Pest---------------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("File:",f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("basepath:",basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("Uploaded!!!",file_path)
        #Make predicitions
        preds = model_predict_pest(file_path)
        print("Preds:",preds)

        lalbel2=preds.max(1)
        print("Labels2: ",lalbel2)
        #pred_class = decode_predictions(preds, top=1) 
        pred=np.where(preds == lalbel2)
        print("pred:",pred)
        id=pred[1][0]
        print("id:",id)
        print(preds)
        id1=str(pred[1])
        #id = str(pred_class[0][0][1]) 
       # print(preds)
        #id1=pred[1]
        return id1

    return None

#--------------------------------------Prediction route for disease-----------------------------------
@app.route('/predict_disease', methods=['GET', 'POST'])
def upload_disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
        # Get the file from post request
        f1 = request.files['file']
        print("File_disease:",f1)
        # Save the file to ./uploads
        basepath_disease = os.path.dirname(__file__)
        print("basepath_of_disease:",basepath_disease)
        file_path_disease = os.path.join(
            basepath_disease, 'upload_disease', secure_filename(f1.filename))
        f1.save(file_path_disease)
        print("Uploaded_disease!!!",file_path_disease)
        prediction=model_predict_disease(file_path_disease)
        print("Prediction for disease:",prediction)

        prediction_dis=prediction[0]
        print(prediction_dis)
        print(max(prediction_dis))
	    #print(preditcion.index(max(preditcion)))
        id= np.where(prediction_dis == max(prediction_dis))
        id=str(id[0][0])
        print("ID:",id)
        return id
    else:
        return render_template("index.html")
    return None





#-----------------------Server----------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)

