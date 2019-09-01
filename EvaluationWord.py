from Parameter import *
from Utils import *
from sklearn.model_selection import train_test_split
import editdistance
from Spell import correction
from keras.models import model_from_json


if __name__=='__main__':
	paths_and_texts = get_paths_and_texts(is_words=True)
	print('number of image: ', len(paths_and_texts))

	paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.4, random_state=1707)
	paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.5, random_state=1707)
	print('number of train image: ', len(paths_and_texts_train))
	print('number of valid image: ', len(paths_and_texts_val))
	print('number of test image: ', len(paths_and_texts_test))

	with open('Resource/word_model_predict.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights('Resource/iam_words--15--1.791.h5')

	ed_chars = num_chars = ed_words = num_words = 0
	for path, gt_text in paths_and_texts_test:
		pred_text = predict_image(model, path, is_word=True)
		if gt_text!=pred_text:
			ed_words += 1 
		num_words += 1
		ed_chars += editdistance.eval(gt_text, pred_text)
		num_chars += len(gt_text)
	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)