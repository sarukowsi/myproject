#CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Configuration ---
app = Flask(__name__)
# Define upload and model folders
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'trained_model.h5'  # Update this to your model's path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model (adjust as needed for your model type)
try:
    # Use load_model for HDF5 (.h5) or SavedModel format
    model = tf.keras.models.load_model(MODEL_PATH)
    # Define class names (important for displaying the result)
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
 # <--- **UPDATE THIS WITH YOUR ACTUAL CLASS NAMES**
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image and make prediction
def model_predict(img_path, model):
    # Adjust target_size to what your CNN model expects (e.g., 224x224)
    target_size = (224, 224) # <--- **UPDATE THIS FOR YOUR MODEL**
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch dimension
    
    # Normalize the image data (adjust if your model uses a different scale)
    # Assuming pixel values need to be scaled to [0, 1]
    img_array = img_array / 255.0  # <--- **ADJUST NORMALIZATION**

    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class index and probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_probability = predictions[0][predicted_class_index]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    return predicted_class_name, predicted_probability


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part in the request.')
        
        file = request.files['file']
        
        # Check if user selected a file
        if file.filename == '':
            return render_template('index.html', error='No file selected.')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Save the file
            file.save(filepath)
            
            # Make prediction
            if model:
                predicted_class, probability = model_predict(filepath, model)
                prediction_result = f"Prediction: {predicted_class} (Confidence: {probability:.2f})"
                
                # Render template with the uploaded image and prediction
                return render_template(
                    'index.html', 
                    prediction=prediction_result, 
                    image_url=url_for('static', filename='uploads/' + filename)
                )
            else:
                return render_template('index.html', error='Model not loaded.')
        else:
            return render_template('index.html', error='File type not allowed. Please upload PNG, JPG, or JPEG.')

    # For a GET request, just show the upload page
    return render_template('index.html')

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)