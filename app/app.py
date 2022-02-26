import base64
from os import environ

import cv2
import flask
import numpy as np
from flask import render_template, request

from Calculate import evaluate
from Predict import predictExpression

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
        predictions = predictExpression(255 - image)
        result, error = evaluate(predictions) if predictions else (None, "")

        if error == "":
            return render_template('results.html', prediction=predictions, result=result)
        else:
            return render_template('results.html', prediction=predictions, error=error)
    else:
        pass


if __name__ == '__main__':
    evaluate(['2','-','2'])
    app.run(host='0.0.0.0', debug=True, port=environ.get("PORT", 5000))
