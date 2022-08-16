# Arabic-Signs-Recognition
This project is a desktop application for arabic hand sign language recognition.

You can dowloand the complete project, or only [GUI, saved_model] folders for the desktop app.

For direct opencv/console test for the tensorflow model you can run the cv2_test.py script.




## Machine learning
Dataset from : https://data.mendeley.com/datasets/y7pckrw6z2/1, CC BY 4.0 license

Trained by the training_scripts/Arabic_Signs_Rec_gray.ipynb google colab notebook

The model is saved in 'saved_model' folder with the '.h5' extention

So far the "ARS_REC_model_gray_v3.h5" is the most accurate model

It has a performance of 94% accuracy

Here is the heatmap of the confusion matrix foreach letter :
![ARS_Heatmap_grayV3](https://user-images.githubusercontent.com/86375309/168066530-10c87c52-76df-41d5-9587-2794018590f9.png)


## Web page
more info in ...
In the web-page folder, the app is developped with :
Reactjs for frontend and backend using flask api to predict images from opencv camera


## Graphical user interface
The GUI folder contains '.py' script for the gui using PyQT5 framework

