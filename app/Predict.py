import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array


def predictExpression(img):
    # load the trained model
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])

    # Pre-process the image to predict
    kernel1 = np.ones((5, 5), np.uint8)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, np.ones((1, 1), np.uint8), iterations=1)
    img = cv2.erode(img, kernel1, iterations=1)
    ret, thresh_img = cv2.threshold(img, 127, 255, 0)

    # find contours to find different symbols in the image
    cnts, hierarchy = cv2.findContours(thresh_img, 3, 2)
    results = []
    cnt_t = []
    k = len(cnts)
    # removing repeated contours for same symbol
    for i, cnt in enumerate(cnts):
        if i != 0 and hierarchy[0][i][3] == 0:
            cnt_t.append(cnt)
    cnts = sorted(cnt_t, key=lambda b: b[0][0][0], reverse=False)

    # Extracting each symbol to predict
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.dilate(img, kernel, iterations=1)
        img_temp = img[y:y + h, x:x + w]
        img_temp = (255 - img_temp)
        top_bottom_padding = int((max(w, h) * 1.2 - h) / 2)
        left_right_padding = int((max(w, h) * 1.2 - w) / 2)
        img_temp = cv2.copyMakeBorder(img_temp, top_bottom_padding, top_bottom_padding, left_right_padding,
                                      left_right_padding,cv2.BORDER_CONSTANT,value=0)
        img_temp = cv2.resize(img_temp, (28, 28))
        img_temp = img_to_array(img_temp)
        img_temp = np.array(img_temp, dtype="float") / 255.0

        pred = loaded_model.predict(img_temp.reshape(1, 28, 28, 1))

        result = pred.argmax()
        label_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                      '8': '8', '9': '9', '10': '+', '11': '-', '12': 'x', '13': '<', '14': '>', '15': '!='}
        results.append(label_dict[str(result)])

    # return the list of predictions of each symbol
    return results
