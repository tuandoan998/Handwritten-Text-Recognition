from Parameter import *
from Utilts import *
from CRNN_Model import CRNN_model
from ImageGenerator import TextImageGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import editdistance

sess = tf.Session()
K.set_session(sess)

if __name__=='__main__':
	paths_and_texts = get_paths_and_texts()
	print('number of image: ', len(paths_and_texts))

	paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.4, random_state=1707)
	paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.5, random_state=1707)
	print('number of train image: ', len(paths_and_texts_train))
	print('number of valid image: ', len(paths_and_texts_val))
	print('number of test image: ', len(paths_and_texts_test))

	model, _ = CRNN_model()
	model.load_weights('iam_words--15--1.791.h5')

	test_set = TextImageGenerator(paths_and_texts_test, 128, 64, 8, 4)
	test_set.build_data()

	net_inp = model.get_layer(name='the_input').input
	net_out = model.get_layer(name='softmax').output

	ed_chars = 0
	num_chars = 0
	ed_words = num_words = 0
	batch = 0
	num_batch = int(test_set.n/32)
	for inp_value, _ in test_set.next_batch():
		if batch>num_batch:
			break
		print('batch: %s/%s' % (batch, str(num_batch)))
		batch = batch+1
		bs = inp_value['the_input'].shape[0]
		X_data = inp_value['the_input']
		net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
		pred_texts = decode_batch(net_out_value)
		labels = inp_value['the_labels']
		label_len = inp_value['label_length']
		g_texts = []
		for label in labels:
			g_text = ''.join(list(map(lambda x: letters[int(x)], label)))
			g_texts.append(g_text)

		for i in range(bs):
			g_texts[i] = g_texts[i][:int(inp_value['label_length'].item(i))]
			ed_chars += editdistance.eval(g_texts[i], pred_texts[i])
			if g_texts[i]!=pred_texts[i]:
				ed_words += 1
			num_chars += len(g_texts[i])
			num_words += 1
	
	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)