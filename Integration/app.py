from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import os
from keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

class_labels = ['apple', 'hourglass', 'rectangle', 'pear', 'inverted_triangle']

model = load_model('model.h5')



# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Function to predict the label of the uploaded image
def predict_label(img_array):
    """Predict the label of the given image array."""
    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        return class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"
    except Exception as e:
        return f"Prediction error: {str(e)}"



# Route for home page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route("/about")
def about_page():
    return "About You..!!!"


# submit
@app.route("/submit", methods=['POST'])
def get_hours():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return "No file part", 400  # Handle missing file part

        img = request.files['my_image']
        if img.filename == '':
            return "No selected file", 400  # Handle no file selected

        # Check if the file is an allowed type
        if not allowed_file(img.filename):
            return "File type not allowed", 400
        
        # Save the image to the static folder for temporary storage
        img_path = os.path.join('static', img.filename)
        img.save(img_path)

        # Load and process the image
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize to model's input size
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for model input

        # Get the prediction
        prediction = predict_label(img_array)

        return render_template("home.html", prediction=prediction, img_path=img_path)


# prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'})
    
    # Process the image
    try:
        img = Image.open(file.stream)
        img = img.resize((224, 224))  # Resize to model's input size
        img_array = np.array(img) / 255.0  # Normalize image
        img_array = img_array.reshape(1, 224, 224, 3)  # Reshape for model input
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
