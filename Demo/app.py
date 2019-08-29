from flask import Flask, request, render_template, send_from_directory
from ocr import detect
from keras.models import model_from_json
import os
from keras import backend as K

app = Flask(__name__)
UPLOAD_FOLDER = './static'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("template.html")

@app.route("/upload", methods=["POST"])
def upload():
    K.clear_session()
    with open('../Resource/model_predict.json', 'r') as f:
        model_predict = model_from_json(f.read())
    model_predict.load_weights('../Resource/iam_words--15--1.791.h5')

    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    text, locate = detect(model_predict, destination)
    K.clear_session()
    pred = ' '.join(text)
    return render_template("template.html", predict=pred, image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
