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

def predict(w_model_predict, l_model_predict, test_img):
	res = []
	text = []
	img = prepareImg(cv2.imread(test_img), 64)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		cv2.imwrite('tmp/%d.png'%j, wordImg)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	for f in imgFiles:
		text.append(predict_image(w_model_predict, 'tmp/'+f, is_word=True))
	shutil.rmtree('tmp')
	text = correction_list(text)
	text1 = ' '.join(text)
	text2 = predict_image(l_model_predict, test_img, is_word=False)
	return text1, text2


if __name__=='__main__':
	with open('../Resource/line_model_predict.json', 'r') as f:
		l_model_predict = model_from_json(f.read())
	with open('../Resource/word_model_predict.json', 'r') as f:
		w_model_predict = model_from_json(f.read())
	w_model_predict.load_weights('../Resource/iam_words--15--1.791.h5')
	l_model_predict.load_weights('../Resource/iam_lines--12--17.373.h5')
	text1, text2 = predict(w_model_predict, l_model_predict, '../Resource/test_img/2.png')
	print('--------------PREDICT---------------')
	print('[Word model]: ', text1)
	print('[Line model]: ', text2)
