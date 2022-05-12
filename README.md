# Arabic-Signs-Recognition
This project is a website and desktop application for arabic hand sign language recognition.

You can dowloand the complete project, or only [GUI, saved_model] folders for the desktop app, or [web_app, saved_model] folders for the reactjs+flaskAPI project.

For direct console opencv test for the tensorflow model you can use the cv2_test.py script.

Please modify the path to the saved model in the utils/Sign_Recognation.py script to the absolute path in your machine (Or your proper tensorflow model h5 file).




## Machine learning
Dataset from : https://data.mendeley.com/datasets/y7pckrw6z2/1, CC BY 4.0 license

Trained by the training_scripts/Arabic_Signs_Rec_gray.ipynb google colab notebook

The model is saved in 'saved_model' folder with the '.h5' extention

It has a performance of 92% accuracy
![ARS_Heatmap_grayV3](https://user-images.githubusercontent.com/86375309/168066530-10c87c52-76df-41d5-9587-2794018590f9.png)


## Web page
In the web-page folder, the app is developped with :
Reactjs for frontend and backend using flask api to predict images from opencv camera


## Graphical user interface
The GUI folder contains .py script for the gui using PyQT5 framework and utils folder for machine learning script

