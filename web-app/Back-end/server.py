from flask import Flask,Response,request
from camera import VideoCamera
import cv2
from utils import Sign_Recognition as sr
import requests

  
# Initializing flask app
app = Flask(__name__)
i = 0
index = 0

camera = VideoCamera()

def gen():
    global index

    while True:
        frame = camera.get_frame()
        cv2.rectangle(frame , (300,300) , (100,100), (0,255,0) , 0)
        crop_img = frame[100:300, 100:300]
        index = sr.predict_img(crop_img)
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')  

# Route for seeing a data
@app.route('/data')
def get_prediction():
    # Returning an api for showing in reactjs

    r_socket = {'pred_num' : i , 'prediction' : sr.CATEGORIES[index]}
    #print('emit socket is : ' , r_socket)
    return r_socket


# Running app
if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000, threaded=True, use_reloader=False)