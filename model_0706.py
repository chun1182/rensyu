from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
import numpy as np
from keras.optimizers import SGD, Adagrad, Adam

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

def VGG16_convolutions():
    #1conv-P => 1conv-P => 1conv-P => 2conv => GAP => dense
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(30,30,3)))
    model.add(Conv2D(16, (3, 3), activation='relu', name='conv1_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(16, (3, 3),padding='same', activation='relu',name='conv2_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(32, (3, 3),padding='same', activation='relu',name='conv3_1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same', activation='relu',name='conv4_1'))
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu', name='conv4_2'))
#    model.add(MaxPooling2D((2, 2), strides=(2, 2))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation = 'softmax' ))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)

    return model

def VGG16_convolutions2():
   #2conv-P => 2conv-P => 2conv => GAP => dense
    
    #K.set_learning_phase() #set learning phase
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(30,30,3)))
    model.add(Conv2D(16, (3,3),activation='relu', name='conv1_1'))
    model.add(Conv2D(16, (3,3),padding='same', activation='relu', name='conv1_2'))
    #model.add(Conv2D(16,(3,3),strides=(2,2),padding='same', activation='relu', kernel_initializer='he_normal', name='conv1a'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3),padding='same', activation='relu', name='conv2_1'))
    model.add(Conv2D(32, (3, 3),padding='same', activation='relu', name='conv2_2'))
    #model.add(Conv2D(32, (3,3),strides=(2,2),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3),padding='same', activation='relu', name='conv3_1'))
    model.add(Conv2D(64, (3, 3),padding='same', activation='relu', name='conv3_2'))
    #model.add(Conv2D(128, (3, 3),padding='same', activation='relu', name='conv3_3'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation = 'softmax' ))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)

    return model

def VGG16_convolutions3():
    #2conv-P => 2dense-dropout => dense
    
    #K.set_learning_phase() #set learning phase
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(30,30,3)))
    model.add(Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', name='conv1_1'))
    model.add(Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_1'))
    model.add(Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu',name='Dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu',name='Dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_initializer='uniform'))

    return model

def VGG16_convolutions4():
    #2conv-P => 2dense-dropout => dense
    
    #K.set_learning_phase() #set learning phase
    model = Sequential()
    model.add(ZeroPadding2D((1, 1),input_shape=(30,30,3)))
    model.add(Conv2D(16,(3,3), activation='relu', kernel_initializer='he_normal', name='conv1_1'))
    model.add(Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv1_2'))
    #model.add(Conv2D(16,(3,3),strides=(2,2),padding='same', activation='relu', kernel_initializer='he_normal', name='conv1a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    
    model.add(Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_1'))
    model.add(Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2_2'))
    #model.add(Conv2D(32,(3,3),strides=(2,2),padding='same', activation='relu', kernel_initializer='he_normal', name='conv2a'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv3_1'))
    model.add(Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal', name='conv3_2'))
    #model.add(Conv2D(64,(3,3),strides=(2,2),padding='same', activation='relu', kernel_initializer='he_normal', name='conv3a'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    
    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', kernel_initializer='uniform'))

    return model

def get_model():
    model = VGG16_convolutions3()
 #   model = load_model("weights.49-0.41.hdf5")
    #model.summary()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
    return model

def load_model_weights(model, weights_path):
    print ('Loading model.')
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = True
    f.close()
    print ('Model loaded.')
    return model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

#random erasing when p% cover s_l~s_h% color v_l~v_h
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser

