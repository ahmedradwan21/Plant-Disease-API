# predictor/services.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

model_path = 'plant disease_98.72.h5' 
model = load_model(model_path)

class_names = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Apple Rust', 'Apple Healthy',
    'Blueberry Healthy', 'Cherry Powdery Mildew', 'Cherry Healthy', 'Corn Gray Leaf Spot',
    'Corn Common Rust', 'Corn Northern Leaf Blight', 'Corn Healthy', 'Grape Black Rot',
    'Grape Esca (Black Measles)', 'Grape Leaf Blight (Isariopsis Leaf Spot)', 'Grape Healthy',
    'Orange Haunglongbing (Citrus Greening)', 'Peach Bacterial Spot', 'Peach Healthy',
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight', 'Potato Late Blight',
    'Potato Healthy', 'Raspberry Healthy', 'Soybean Healthy', 'Squash Powdery Mildew',
    'Strawberry Leaf Scorch', 'Strawberry Healthy', 'Tomato Bacterial Spot', 'Tomato Early Blight',
    'Tomato Late Blight', 'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
    'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Tomato Healthy',
    'Apple Bitter Rot', 'Apple Alternaria Leaf Spot', 'Apple Sooty Blotch', 'Blueberry Anthracnose',
    'Cherry Brown Rot', 'Corn Downy Mildew', 'Corn Anthracnose Leaf Blight', 'Grape Anthracnose',
    'Orange Black Spot', 'Peach Leaf Curl', 'Pepper Anthracnose', 'Potato Black Scurf',
    'Raspberry Anthracnose', 'Soybean Rust', 'Squash Downy Mildew', 'Strawberry Powdery Mildew',
    'Tomato Gray Mold', 'Tomato Fusarium Wilt', 'Tomato Bacterial Canker', 'Apple Fly Speck',
    'Blueberry Mummy Berry', 'Cherry Bacterial Canker', 'Corn Stalk Rot', 'Grape Powdery Mildew',
    'Orange Alternaria Brown Spot', 'Peach Scab', 'Pepper Bacterial Wilt', 'Potato Pink Rot',
    'Raspberry Spur Blight', 'Soybean Frog Eye Leaf Spot', 'Squash Bacterial Wilt', 'Strawberry Gray Mold',
    'Tomato Verticillium Wilt', 'Tomato Bacterial Wilt', 'Tomato Powdery Mildew', 'Tomato Anthracnose',
    'Apple Fire Blight', 'Blueberry Stem Blight', 'Cherry Leaf Spot', 'Corn Smut', 'Grape Downy Mildew',
    'Orange Canker', 'Peach Leaf Rust', 'Pepper Cercospora Leaf Spot', 'Potato Pythium Root Rot',
    'Raspberry Yellow Rust', 'Soybean Downy Mildew', 'Squash Fusarium Wilt', 'Strawberry Verticillium Wilt',
    'Tomato Bacterial Spot', 'Tomato Southern Blight', 'Tomato Early Blight', 'Tomato Late Blight',
    'Apple Collar Rot', 'Blueberry Rust', 'Cherry Shot Hole', 'Corn Southern Rust', 'Grape Botrytis Bunch Rot',
    'Orange Greasy Spot', 'Peach Powdery Mildew', 'Pepper Verticillium Wilt', 'Potato Fusarium Dry Rot',
    'Raspberry Botrytis Fruit Rot', 'Soybean Charcoal Rot', 'Squash Anthracnose', 'Strawberry Angular Leaf Spot',
    'Tomato White Mold',
]

def prepare_image(img_path, target_size=(200, 200)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    img_array = prepare_image(img_path)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)
    result = class_names[pred_class[0]]
    return result
