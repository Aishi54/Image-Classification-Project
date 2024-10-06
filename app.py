from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('wild_animals_model.h5')

# Define the class labels
class_labels = ['Cheetah', 'Lion']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    data = request.json
    image_data = data['image']
    
    # Decode the image
    image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make a prediction
    prediction = model.predict(image)
    label = class_labels[np.argmax(prediction)]
    
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
