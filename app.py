import flask
from flask import Flask, render_template, request
import base64
import numpy as np
import cv2
from Predict import Predictit, Calculate
import json

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21

# Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')


# First route : Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')


# Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Preprocess the image
        # Access the image
        draw = request.form['url']
        # Removing the useless part of the url.
        draw = draw[init_Base64:]
        # Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        predictions = Predictit(255-image)
        print(predictions)
        result, error = Calculate(predictions)

        if error == "":
            return render_template('results.html', prediction=predictions, result=result)
        else:
            return render_template('results.html', prediction=predictions, error=error)
    else:
        pass


if __name__ == '__main__':
    app.run(debug=True)
