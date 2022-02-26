# [HandwritingBasedCalculator](https://handwritting-based-calculator.herokuapp.com/)

A Calculator Web App that recognizes handwritten numbers and math symbols and can perform arithmetic and relational operations.

Based on a CNN model that classifies the symbol into 0:9, +, -, x, >, < and neq after extracting it from the input image using image processing techniques.

Deployed to: https://handwritting-based-calculator.herokuapp.com/
### Libraries Used:
* [Keras](https://keras.io/) : for building and training a CNN model
* [OpenCV](https://opencv.org/) : for image processing tasks
* [Flask](https://flask.palletsprojects.com/) : for deploying as a web app

### App in Action:

 - can recognize multi-digit numbers and perform arithmetic operations:
![](app1.gif)

- can perform relational operations and gives result as True or False: 
![](app3.gif)

- can check if the syntax is correct or not: 
![](app2.gif)

### WorkFlow:
- Data acquisition: [Dataset from Kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols/)
- Data cleaning and processing: cleaned data for required symbols to data_cleaned.rar (extracted) 
- Data Modelling: Built and Trained CNN model using Keras
- Input image pre-processed using OpenCV for basic image processing and to extracted contours to get individual symbols
- Each symbol predicted by giving extracted symbol image as an input to the model
- Mathematical operations are performed on the predictions and output a final result
- Web Application: Used Flask to envelope the model as a web app to provide the user a simple interface
- Deployment: Deployed the Web App to Heroku using Docker

### Steps to run the application locally:
- Install Requirements:
  - Python 3.8+ (preferred 3.10.2)
  - For other requirements, run
    ```pip install -r Requirements.txt```
    (In case of installation failures due to opencv, try: ```apt-get update && apt-get install -y python3-opencv```.) 
- Clone the GitHub Repository and cd into project's root directory.
- Run the application using
    ```python app.py```
  . Application should now run on http://<span>localhost:5000/</span>

### Todos:
 - Add more symbols like divide, log, square root, etc.
 - Improve web app, make more dynamic and responsive
 - Add functionalities to the drawing canvas like resize, eraser, etc.

#### Last Release Updates:
 - Deployed to Heroku using Docker
 - Refactored and Updated code
