import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h-old_h)/2), int((new_h-old_h)/2)+old_h
    w1, w2 = int((new_w-old_w)/2), int((new_w-old_w)/2)+old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2,:] = img
    return img_pad

def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w<target_w and h<target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w>=target_w and h<target_h:
        new_w = target_w
        new_h = int(h*new_w/w)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w<target_w and h>=target_h:
        new_h = target_h
        new_w = int(w*new_h/h)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        '''w>=target_w and h>=target_h '''
        ratio = max(w/target_w, h/target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    img = cv.imread(path)
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)
    img /= 255
    return img


if __name__=='__main__':
    img = cv.imread('../IAM_lines/lines/a01/a01-000u/a01-000u-00.png', 0)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = preprocess('../IAM_lines/lines/a01/a01-000u/a01-000u-00.png', 800, 64)
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()