from flask import Flask,render_template, Response, request
from flask_ngrok import run_with_ngrok
import cv2
import os
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
from VideoCapture import VideoCapture


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 3024 * 3024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
@app.route('/')
def Home():
    return render_template('/index.html',
                           is_uploaded=False,
                           img_path='',
                           img_size='',
                           img_filename='',
                           img_width=0,
                           img_height=0,
                           img_upload_isvalid=False,
                           predicted_image='',
                           is_predicted=False,
                           is_error=False,
                           is_error_msg=''
                           )

def gen(cam):
    while True:
        ret, frame = cam.get_frame()

        if not ret:
            break

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame
              + b'\r\n\r\n')

@app.route('/video_capture')
def video_capture():
    return Response(
        gen(VideoCapture()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def PIL_bytes(objects):
    objects = objects
    for obj in objects:
        face = obj['face-img']
        buffered = io.BytesIO()
        buffered.seek(0)
        face = face.resize((150, 150))
        face.save(buffered, format="JPEG")
        face_string_encoded = base64.b64encode(buffered.getvalue())
        face_string = face_string_encoded.decode('utf-8')
        obj['face-img'] = face_string
    return objects


@app.route('/fetch_upload',methods=['POST'])
def fetch_upload():
    isValid = False
    image_IO = None
    if request.method == 'POST':
        try:
            img = request.files['img-file']
            img_size = len(img.read())
            img.seek(0)
            image_string_encoded = base64.b64encode(img.read())
            image_string = image_string_encoded.decode('utf-8')
            img.seek(0)
            imgIO = io.BytesIO(img.stream.read())
            imgIO.seek(0)

            image_IO = Image.open(imgIO)
            img_width = image_IO.size[0]
            img_height = image_IO.size[1]
            buffered = io.BytesIO()
            buffered.seek(0)

            image_IO.save(buffered, format="JPEG")
            image_string_encoded = base64.b64encode(buffered.getvalue())
            image_string = image_string_encoded.decode('utf-8')
            imgIO.seek(0)

            # set img display screen to resize the display
            imgIO_display = image_IO
            buffered2 = io.BytesIO()
            buffered2.seek(0)

            imgIO_display = imgIO_display.resize((400,400))
            imgIO_display.save(buffered2, format="JPEG")
            img_display_string_encoded = base64.b64encode(buffered2.getvalue())
            img_display_string = img_display_string_encoded.decode('utf-8')
            imgIO_display.seek(0)

            if int(img_width) > 500 and int(img_height) > 500:
                isValid = True
            return render_template('index.html',
                                                              is_uploaded=True,
                                                              img_display=img_display_string,
                                                              img_path=image_string,
                                                              encoded_img=image_string_encoded,
                                                              img_size=img_size,
                                                              img_filename=img.filename,
                                                              img_width=img_width,
                                                              img_height=img_height,
                                                              img_upload_isvalid=isValid,
                                                              predicted_image='',
                                                              is_predicted=False,
                                                              is_error=False
                                   )


        except OSError as e:
            msg = "Invalid Image - " + str(e)
            return render_template('index.html',
                                   is_uploaded=False,
                                   img_path='',
                                   img_size='',
                                   img_filename='',
                                   img_width=0,
                                   img_height=0,
                                   img_upload_isvalid=False,
                                   predicted_image='',
                                   is_predicted=False,
                                   is_error=True,
                                   is_error_msg=msg
                                   )

@app.route('/predict_capture',methods=["POST"])
def predict_capture():
    if request.method == 'POST':
        base64_str = request.form['img_predict_name']
        #img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
        #img.save('my-image.jpeg')
        img = plt.imread(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))), 0)
        cap = VideoCapture()
        predicted_image, num_faces, list_faces = cap.predict_image(img)
        #predicted_image.save('helloworld.jpg')
        list_faces_string = PIL_bytes(list_faces)
        buffered = io.BytesIO()
        predicted_image = predicted_image.resize((400, 400))
        predicted_image.save(buffered, format="JPEG")
        image_string_encoded = base64.b64encode(buffered.getvalue())
        image_string = image_string_encoded.decode('utf-8')
        if num_faces > 0:
            return render_template('index.html',
                                   is_uploaded=True,
                                   predicted_image=image_string,
                                   is_predicted=True,
                                   is_error=False,
                                   list_emotions=list_faces_string
                                   )
        else:
            msg = "-Theres no Face/s detected in that image. Please upload image that fit to the critera"
            return render_template('index.html',
                                   is_uploaded=False,
                                   img_path='',
                                   img_size='',
                                   img_filename='',
                                   img_width=0,
                                   img_height=0,
                                   img_upload_isvalid=False,
                                   predicted_image='',
                                   is_predicted=False,
                                   is_error=True,
                                   is_error_msg=msg
                                   )



if __name__ == '__main__' :
   app.run(debug=True)