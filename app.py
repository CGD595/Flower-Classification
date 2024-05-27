from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import cv2
from PIL import Image
from img2vec_pytorch import Img2Vec

app = Flask(__name__)

model = joblib.load('svm_flower_classifier.pkl')

img2vec = Img2Vec(model='resnet-18')

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image, (28, 28))
    image_normalized = image_resized / 255.0
    image_flattened = image_normalized.flatten()
    return image_flattened

def extract_features(image):
    image_pil = Image.fromarray(image)
    features = img2vec.get_vec(image_pil)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    features = extract_features(image).reshape(1, -1)
    prediction = model.predict(features)
    return redirect(url_for('output', value=int(prediction[0])))

@app.route('/output')
def output():
    value = request.args.get('value', default=-1, type=int)
    return render_template('output.html', value=value)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
