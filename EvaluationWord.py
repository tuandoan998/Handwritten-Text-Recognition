from Parameter import *
from Utils import *
from sklearn.model_selection import train_test_split
import editdistance
from Spell import correction
from keras.models import model_from_json
from ImageGenerator import TextImageGenerator

if __name__=='__main__':
	test_data = get_paths_and_texts('data/IAM/splits/test.uttlist', is_words=True)
	print('number of test image: ', len(test_data))

	with open('Resource/word_model_predict.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights('Resource/iam_words--20--1.425.h5')

	ed_chars = num_chars = ed_words = num_words = 0
	for path, gt_text in test_data:
		pred_text = predict_image(model, path, is_word=True)
		if gt_text!=pred_text:
			ed_words += 1 
		num_words += 1
		ed_chars += editdistance.eval(gt_text, pred_text)
		num_chars += len(gt_text)
	# batch_size = word_cfg['batch_size']
	# test_set = TextImageGenerator(test_data, word_cfg['img_w'], word_cfg['img_h'], batch_size, word_cfg['input_length'], word_cfg['max_text_len'])
	# print('Loading data for evaluation ...')
	# test_set.build_data()
	# print('Done')
	# print("Number test set: ", test_set.n)

	# batch = 0
	# num_batch = int(test_set.n/batch_size)
	# for inp_value, _ in test_set.next_batch():
	# 	if batch>=num_batch:
	# 		break
	# 	print('batch: %s/%s' % (batch, str(num_batch)))

	# 	labels = inp_value['the_labels']
	# 	label_len = inp_value['label_length']
	# 	g_texts = []
	# 	for label in labels:
	# 		g_text = ''.join(list(map(lambda x: letters[int(x)], label)))
	# 		g_texts.append(g_text)
	# 	pred_texts = decode_batch(model.predict(inp_value))

	# 	for i in range(batch_size):
	# 		g_texts[i] = g_texts[i][:int(inp_value['label_length'].item(i))]
	# 		ed_chars += editdistance.eval(g_texts[i], pred_texts[i])
	# 		num_chars += len(g_texts[i])
	# 		if g_texts[i]!=pred_texts[i]:
	# 			ed_words += 1 
	# 		num_words += 1
 			# batch += 1

		print('ED chars: ', ed_chars)
		print('ED words: ', ed_words)

	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)