from flask import Flask, render_template
from ocr import detect
from keras.models import model_from_json
import os

app = Flask(__name__)
UPLOAD_FOLDER = './static'

@app.route("/")
def home():
    with open('../model_predict.json', 'r') as f:
        model_predict = model_from_json(f.read())
    model_predict.load_weights('../iam_words--15--1.791.h5')
    text, locate = detect(model_predict, '../test_img/4.png')
    pred = ' '.join(text)
    img = os.path.join(UPLOAD_FOLDER, 'summary.png')
    return render_template("template.html", predict=pred, summary_img=img)

if __name__ == "__main__":
    app.run(debug=True)
