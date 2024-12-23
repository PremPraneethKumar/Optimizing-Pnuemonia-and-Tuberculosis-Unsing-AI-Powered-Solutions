from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

# Load the model
model = load_model('Project.h5')  # Updated to use the uploaded Project.h5 model

# Define class labels
class_labels = ['Normal', 'Pneumonia', 'Tuberculosis']

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('Index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['image']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(224, 224))  # Update target_size if model expects different dimensions
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        return render_template('Predict.html', predicted_class=predicted_class, image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)
