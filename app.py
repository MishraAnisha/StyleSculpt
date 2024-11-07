from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, template_folder='template')  # Adjust the path as needed

# Mapping for prediction labels
dic = {0: 'apple', 1: 'hourglass', 2: 'inverted triangle', 3: 'pear', 4: 'rectangle'}

# Load the model
model = load_model('savedmodel.keras')

def predict_label(img_path):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    # Predict and retrieve class index
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]  # Get class with highest probability
    return dic.get(class_index, "Unknown")

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        
        # Save image to static folder
        img_path = os.path.join("static", img.filename)
        img.save(img_path)

        # Predict the class
        prediction = predict_label(img_path)

        return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)

