import numpy as np
from Utils import *
from WordSegmentation import wordSegmentation, prepareImg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Preprocessor import preprocess
from keras.models import model_from_json
import shutil
from keras import backend as K
from keras.utils import plot_model
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
'''
    fig=plt.figure()
    fig.add_subplot(2,1,1)
    test_img=cv2.imread(path)
    plt.imshow(test_img)
    fig.add_subplot(2,1,2)
    plt.imshow(net_out_value.T.squeeze(), cmap='binary', interpolation='nearest')
    plt.yticks(list(range(len(letters) + 1)), letters + ['blank'])
    plt.show()
'''


if __name__=='__main__':
	#model, model_predict = CRNN_model()
	#with open('model_predict.json', 'w') as f:
	#	f.write(model_predict.to_json())
	with open('Resource/model_predict.json', 'r') as f:
		model_predict = model_from_json(f.read())
	#plot_model(model_predict, to_file='model.png', show_shapes=True, show_layer_names=True)
	model_predict.load_weights('Resource/iam_words--15--1.791.h5')

	test_img = 'Resource/test_img/2.jpg'
	
	img = prepareImg(cv2.imread(test_img), 64)
	img2 = img.copy()
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')

	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('tmp/%d.png'%j, wordImg)
		cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image

	cv2.imwrite('Resource/summary.png', img2)
	plt.imshow(img2)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	pred_line = []
	for f in imgFiles:
		pred_line.append(pred_word(model_predict, 'tmp/'+f))
	print('Predict: '+' '.join(pred_line))
	pred_line = correction_list(pred_line)
	print('Predict with spell: '+' '.join(pred_line))

	plt.show()
	shutil.rmtree('tmp')
