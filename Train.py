from Parameter import *
from ImageGenerator import TextImageGenerator
from CRNN_Model import word_model, line_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Utils import *


def train(train_list, val_list, is_word_model):
    if is_word_model:
        model, _ = word_model()
        cfg = word_cfg
    else:
        model, _ = line_model()
        cfg = line_cfg

    input_length = cfg['input_length']
    model_name = cfg['model_name']
    max_text_len = cfg['max_text_len']
    img_w = cfg['img_w']
    img_h = cfg['img_h']
    batch_size = cfg['batch_size']
    train_set = TextImageGenerator(train_list, img_w, img_h, batch_size, input_length, max_text_len)
    print('Loading data for train ...')
    train_set.build_data()
    val_set = TextImageGenerator(val_list, img_w, img_h, batch_size, input_length, max_text_len)
    val_set.build_data()
    print('Done')
    
    print("Number train set: ", train_set.n)
    print("Number val set: ", val_set.n)
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    ckp = ModelCheckpoint(
        filepath=model_name+'--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set.next_batch(),
                        steps_per_epoch=train_set.n // batch_size,
                        epochs=32,
                        validation_data=val_set.next_batch(),
                        validation_steps=val_set.n // batch_size,
                        callbacks=[ckp, earlystop])

    return model

if __name__=='__main__':
    paths_and_texts = get_paths_and_texts(is_words=True)
    print('number of image: ', len(paths_and_texts))

    paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.4, random_state=1707)
    paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.5, random_state=1707)
    print('number of train image: ', len(paths_and_texts_train))
    print('number of valid image: ', len(paths_and_texts_val))
    print('number of test image: ', len(paths_and_texts_test))

    model = train(paths_and_texts_train, paths_and_texts_val, True)