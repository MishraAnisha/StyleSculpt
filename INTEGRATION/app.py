from flask import Flask, render_template, request, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/images'

# Mapping for prediction labels
dic = {0: 'apple', 1: 'hourglass', 2: 'inverted triangle', 3: 'pear', 4: 'rectangle'}

# Load the pre-trained Keras model
model = load_model('savedmodel.keras')

def predict_label(img_path):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict and retrieve class index
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)[0]
    return dic.get(class_index, "Unknown")

@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/recom")
def recommendation():
    return render_template("recom.html")

@app.route("/cloth")
def cloth():
    body_shape = request.args.get('body_shape', default='apple')  # Get body shape from the query parameter
    return render_template("cloth.html", body_shape=body_shape)

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        prediction = predict_label(img_path)

        # Send prediction and img_path to the template for rendering
        return render_template("index.html", prediction=prediction, img_path=img.filename)

@app.route("/know_more")
def know_more():
    return render_template("know.html")

# Route for the "Challenge" page
@app.route("/challenge")
def challenge():
    return render_template("challenge.html")

if __name__ == '__main__':
    app.run(debug=True)
