from keras.models import model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from operations import do_operations
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import cv2
import imutils
import math


def Predictit(img):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])

    # img = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)

    # cv2.imshow('img', img)
    kernel1 = np.ones((5, 5), np.uint8)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, np.ones((1, 1), np.uint8), iterations=1)
    img = cv2.erode(img, kernel1, iterations=1)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    # cv2.imshow('img', thresh)
    # cv2.waitKey(0)
    cnts, hierarchy = cv2.findContours(thresh, 1, 2)
    i = 0
    cnts = cnts[:-1]
    cnts = sorted(cnts, key=lambda b: b[0][0][0], reverse=False)
    # print(len(cnts))
    results = []
    cnt_t = []
    k = len(cnts)
    # print(k)
    for z in range(k):
        cnt_temp = sorted(cnts[z], key=lambda b: b[0][0], reverse=False)
        cnt_t.append(cnt_temp[0][0][0])
    z = 0

    while z < k:
        if (z >= 1):
            if (cnt_t[z] - cnt_t[z-1]) < 0.1*max(thresh.shape):
                if cnt_t[z] < cnt_t[z-1]:
                    x = z-1
                else:
                    x = z
                cnts.remove(cnts[x])
                cnt_t.remove(cnt_t[x])
                z = z-1
        k = len(cnts)
        z = z+1

    # print("cnts: ", len(cnts))
    for cnt in cnts:
        # print(len(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        # print(x, " ", y, " ", w, " ", h)
        img = cv2.dilate(img, kernel, iterations=1)
        img_temp = img[y:y+h, x:x+w]
        # img_temp = img[y-int(0.05*h):y+int(1.05*h), x -
        #                int(0.05*w):x+int(1.05*w)]
        img_temp = (255 - img_temp)
        if w < 0.2*h:
            white = np.zeros((h, math.floor(2.5*w)))
            # print(white.shape, " ", img_temp.shape)
            img_temp = np.concatenate((white, img_temp, white), axis=1)
            h, w = img_temp.shape
        if h < 0.2*w:
            white = np.zeros((math.floor(2.5*h), w))
            # print(white.shape, " ", img_temp.shape)
            img_temp = np.concatenate((white, img_temp, white), axis=0)
            h, w = img_temp.shape

        img_temp = cv2.resize(img_temp, (28, 28))
        img_temp = img_to_array(img_temp)
        img_temp = np.array(img_temp, dtype="float")/255.0

        # cv2.imshow('img' + str(i), img_temp)
        # cv2.waitKey(0)
        pred = loaded_model.predict(img_temp.reshape(1, 28, 28, 1))

        result = pred.argmax()
        label_dict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
                      '8': '8', '9': '9', '10': '+', '11': '-', '12': 'x', '13': '<', '14': '>', '15': '!='}
        results.append(label_dict[str(result)])

    # print(results)
    return results


def Calculate(predictions):

    label_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                  '8': 8, '9': 9, '+': '+', '-': '-', 'x': 't', '<': 'l', '>': 'g', '!=': 'n'}
    results = [label_dict[pred] for pred in predictions]
    k = len(results)
    if results[0] not in range(0, 10):
        return (float('nan'), "Invalid Syntax")
    i = 1
    while i < k:
        if results[i] in range(0, 10) and results[i-1] in range(0, 10):
            results[i] = results[i-1]*10+results[i]
            results.remove(results[i-1])
            k -= 1
        else:
            i += 1
    # print(results)
    k = len(results)

    # Check for relational operations
    is_relational = False
    for i, result in enumerate(results):
        if result in ['l', 'g', 'n']:
            if not is_relational:
                is_relational = True
                result1, error1 = do_operations(results[:i])
                result2, error2 = do_operations(results[i+1:])
                if (result1 and result2 and error1 == "" and error2 == ""):
                    if result == 'l':
                        return ("True" if result1 < result2 else "False", error1+error2)
                    elif result == 'g':
                        return ("True" if result1 > result2 else "False", error1+error2)
                    elif result == 'n':
                        return ("True" if result1 != result2 else "False", error1+error2)
                else:
                    return (float('nan'), "Invalid Syntax")
            else:
                return (float('nan'), "Invalid Syntax")

    # print(is_relational)
    if not is_relational:
        result, error = do_operations(results)
    if not math.isnan(result):
        return (result, error)
    # cv2.waitKey(0)
