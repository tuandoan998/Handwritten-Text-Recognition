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
from ImageGenerator import TextImageGenerator

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

	batch_size = line_cfg['batch_size']
	test_set = TextImageGenerator(paths_and_texts_test, line_cfg['img_w'], line_cfg['img_h'], batch_size, line_cfg['input_length'], line_cfg['max_text_len'])
	print('Loading data for evaluation ...')
	test_set.build_data()
	print('Done')
	print("Number test set: ", test_set.n)

	ed_chars = num_chars = ed_words = num_words = 0
	batch = 0
	num_batch = int(test_set.n/batch_size)
	for inp_value, _ in test_set.next_batch():
		if batch>=num_batch:
			break
		print('batch: %s/%s' % (batch, str(num_batch)))

		labels = inp_value['the_labels']
		label_len = inp_value['label_length']
		g_texts = []
		for label in labels:
			g_text = ''.join(list(map(lambda x: letters[int(x)], label)))
			g_texts.append(g_text)
		pred_texts = decode_batch(model_predict.predict(inp_value))

		for i in range(batch_size):
			g_texts[i] = g_texts[i][:int(inp_value['label_length'].item(i))]
			ed_chars += editdistance.eval(g_texts[i], pred_texts[i])
			num_chars += len(g_texts[i])
			(idStrGt, idStrPred) = getWordIDStrings(g_texts[i], pred_texts[i])
			ed_words += editdistance.eval(idStrGt, idStrPred)
			num_words += len(idStrGt)

		print('ED chars: ', ed_chars)
		print('ED words: ', ed_words)
		batch += 1
	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)
