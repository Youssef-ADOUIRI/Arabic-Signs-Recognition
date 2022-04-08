from flask import Flask,Response,render_template
import datetime
from camera import VideoCamera as cam

  
x = datetime.datetime.now()
  
# Initializing flask app
app = Flask(__name__)


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(cam()),mimetype='multipart/x-mixed-replace; boundary=frame')  

# Route for seeing a data
@app.route('/data')
def get_time():
    # Returning an api for showing in  reactjs
    return {
        'Name':"Arabic signs", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }
  
      
# Running app
if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0', port=5000, threaded=True, use_reloader=False)