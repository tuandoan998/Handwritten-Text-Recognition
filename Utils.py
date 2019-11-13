import os
import numpy as np
import itertools
from Parameter import *
from Preprocessor import preprocess
from keras import backend as K


def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(letters):
            outstr += letters[c]
    return outstr

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

def get_paths_and_texts(is_words):
    paths_and_texts = []
    if is_words:
        with open('../IAM_words/words.txt') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err':
                    continue
                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join('../IAM_words/words', label_dir, sub_label_dir, fn)
                gt_text = ' '.join(line_split[8:])
                paths_and_texts.append([img_path, gt_text])
    else:
        with open('../IAM_lines/lines.txt') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err':
                    continue
                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join('../IAM_lines/lines', label_dir, sub_label_dir, fn)
                gt_text = ' '.join(line_split[8:])
                gt_text = gt_text.replace('|', ' ')
                l = len(gt_text)
                if l<10 or l>74:
                    continue
                paths_and_texts.append([img_path, gt_text])
    return paths_and_texts

def predict_image(model_predict, path, is_word):
    if is_word:
        width = word_cfg['img_w']
    else:
        width = line_cfg['img_w']
    img = preprocess(path, width, 64)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decode_label(net_out_value)
    return pred_texts
