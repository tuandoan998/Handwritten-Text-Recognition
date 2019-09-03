from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import editdistance
import os
from WordSegmentation import wordSegmentation, prepareImg
import cv2
import numpy as np
from Preprocessor import preprocess
from Utils import get_paths_and_texts
from keras import backend as K
from Utils import *
import shutil
import re
from Spell import correction_list

pattern = '[' + r'\w' + ']+'
def getWordIDStrings(s1, s2):
	# get words in string 1 and string 2
	words1 = re.findall(pattern, s1)
	words2 = re.findall(pattern, s2)
	# find unique words
	allWords = list(set(words1 + words2))
	# list of word ids for string 1
	idStr1 = []
	for w in words1:
		idStr1.append(allWords.index(w))
	# list of word ids for string 2
	idStr2 = []
	for w in words2:
		idStr2.append(allWords.index(w))
	return (idStr1, idStr2)

def detect_word_model(model_predict, test_img):
	img = prepareImg(cv2.imread(test_img), 64)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		cv2.imwrite('tmp/%d.png'%j, wordImg)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	pred_line = []
	for f in imgFiles:
		pred_line.append(predict_image(model_predict, 'tmp/'+f, True))
	shutil.rmtree('tmp')
	pred_line = correction_list(pred_line)
	return (' '.join(pred_line))

if __name__=='__main__':
	paths_and_texts = get_paths_and_texts(is_words=False)
	print('number of image: ', len(paths_and_texts))

	paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.3, random_state=1707)
	paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.65, random_state=1707)
	print('number of train image: ', len(paths_and_texts_train))
	print('number of valid image: ', len(paths_and_texts_val))
	print('number of test image: ', len(paths_and_texts_test))

	#with open('Resource/word_model_predict.json', 'r') as f:
	#	model_predict = model_from_json(f.read())
	#model_predict.load_weights('Resource/iam_words--15--1.791.h5')
	with open('Resource/line_model_predict.json', 'r') as f:
		model_predict = model_from_json(f.read())
	model_predict.load_weights('Resource/iam_lines--12--17.373.h5')

	ed_chars = num_chars = ed_words = num_words = 0
	for path, gt_text in paths_and_texts_test:
		#pred_text = detect(model_predict, path)
		#pred_text = predict_image(model_predict, path, False)
		pred_text = predict_image(model_predict, path)
		(idStrGt, idStrPred) = getWordIDStrings(gt_text, pred_text)
		ed_words += editdistance.eval(idStrGt, idStrPred)
		num_words += len(idStrGt)
		ed_chars += editdistance.eval(gt_text, pred_text)
		num_chars += len(gt_text)
	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)
