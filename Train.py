from Parameter import *
from ImageGenerator import TextImageGenerator
from CRNN_Model import CRNN_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from Utils import *


def train(train_list, val_list):
    pool_size = 2
    batch_size = 32
    downsample_factor = pool_size ** 2
    train_set = TextImageGenerator(train_list, img_w, img_h, batch_size, downsample_factor)
    print('Loading data for train ...')
    train_set.build_data()
    val_set = TextImageGenerator(val_list, img_w, img_h, batch_size, downsample_factor)
    val_set.build_data()
    print('Done')
    
    no_train_set = train_set.n
    no_val_set = val_set.n
    print("Number train set: ", no_train_set)
    print("Number val set: ", no_val_set)

    model, y_func = CRNN_model()
    
    try:
        model.load_weights('/content/iam_words--15--1.792.h5')
        print("...Previous weight data...")
    except:
        print("...New weight data...")
        pass
    
    #ada = Adadelta()
    #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    #adam = Adam()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    ckp = ModelCheckpoint(
        filepath='iam_words--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set.next_batch(),
                        steps_per_epoch=no_train_set // batch_size,
                        epochs=32,
                        validation_data=val_set.next_batch(),
                        validation_steps=no_val_set // batch_size,
                        callbacks=[ckp, earlystop])

    return model, y_func

if __name__=='__main__':
    paths_and_texts = get_paths_and_texts()
    print('number of image: ', len(paths_and_texts))

    paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.4, random_state=1707)
    paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.5, random_state=1707)
    print('number of train image: ', len(paths_and_texts_train))
    print('number of valid image: ', len(paths_and_texts_val))
    print('number of test image: ', len(paths_and_texts_test))

    model, model_pred = train(paths_and_texts_train, paths_and_texts_val)