from Parameter import *
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.layers.merge import add, concatenate

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def word_model():
    img_w = word_cfg['img_w']
    img_h = word_cfg['img_h']
    max_text_len = word_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    # Make Networkw
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
    gru1_merged = BatchNormalization()(gru1_merged)
    
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
    gru2_merged = BatchNormalization()(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(gru2_merged) #(None, 32, 80)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=input_data, outputs=y_pred)
    model_predict.summary()

    return model, model_predict


def line_model():
    img_w = line_cfg['img_w']
    img_h = line_cfg['img_h']
    max_text_len = line_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    # Make Networkw
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 800, 64, 1)

    # Convolution layer
    inner = Conv2D(64, (5, 5), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  # (None, 800, 64, 64)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,400, 32, 64)

    inner = Conv2D(128, (5, 5), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 400, 32, 128)
    inner = Activation('relu')(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 400, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 200, 16, 128)
    
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 256)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(256, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 256)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(512, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(512, (3, 3), padding='same', name='conv7', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 512)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max3')(inner)  # (None, 100, 8, 512)

    # CNN to RNN
    inner = Reshape(target_shape=((100, 4096)), name='reshape')(inner)  # (None, 100, 4096)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 100, 64)

    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 100, 512)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 100, 512)
    gru1_merged = BatchNormalization()(gru1_merged)
    
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 100, 1024)
    gru2_merged = BatchNormalization()(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(80, kernel_initializer='he_normal',name='dense2')(gru2_merged) #(None, 100, 80)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=input_data, outputs=y_pred)
    model_predict.summary()

    return model, model_predict