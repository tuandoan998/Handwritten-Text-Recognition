import sys
sys.path.insert(0,'..')

import numpy as np
import numpy as np
import cv2
from keras.models import model_from_json
import shutil
from keras import backend as K
from Utils import *
from WordSegmentation import wordSegmentation, prepareImg
from Preprocessor import preprocess
from Spell import correction_list


def pred_word(model_predict, path):
    img = preprocess(path)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decode_label(net_out_value)
    return pred_texts

def predict(model_predict, test_img):
	res = []
	locate = []
	text = []
	img = prepareImg(cv2.imread(test_img), 64)
	#img2 = img.copy()
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('tmp/%d.png'%j, wordImg)
		locate.append((x, y, w, h))
		#cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image

	#cv2.imwrite('./static/summary.png', img2)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	for f in imgFiles:
		text.append(pred_word(model_predict, 'tmp/'+f))
	shutil.rmtree('tmp')
	text = correction_list(text)
	return text, locate


if __name__=='__main__':
	with open('../Resource/model_predict.json', 'r') as f:
		model_predict = model_from_json(f.read())
	model_predict.load_weights('../Resource/iam_words--15--1.791.h5')
	text, locate = predict(model_predict, '../Resource/test_img/2.png')
	for i in range(len(text)):
		print('Predict: ', text[i])
		print('Locate: ', locate[i])
