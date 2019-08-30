from flask import Flask, request, render_template, send_from_directory
from ocr import predict
from keras.models import model_from_json
import os
import tornado.wsgi
import tornado.httpserver

app = Flask(__name__)
REPO_DIRNAME = os.path.dirname(os.path.abspath(__file__))


class ImagenetClassifier(object):
    def __init__(self):
        print('LOAD MODEL ...')
        with open('../Resource/model_predict.json', 'r') as f:
            self.model_predict = model_from_json(f.read())
        self.model_predict.load_weights('../Resource/iam_words--15--1.791.h5')
    def predict_image(self, image_filename):
        try:
            pred_text, locate = predict(self.model_predict, image_filename)
            return pred_text, locate
        except Exception as err:
            print('Prediction error: %s', err)
            return (False, 'Something went wrong when predict the '
                           'image. Maybe try another one?')

global ocr_model
ocr_model = ImagenetClassifier()

@app.route("/")
def index():
    return render_template("template.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(REPO_DIRNAME, 'images')
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    pred, locate = ocr_model.predict_image(destination)
    text = ' '.join(pred)
    return render_template("template.html", predict=text, image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    #app.run(debug=True)
    start_tornado(app)
