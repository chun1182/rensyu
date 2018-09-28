# -*- coding: utf-8 -*-
#https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py
#MIT License
#Copyright (c) 2016 Jacob Gildenblat

#from keras.applications.vgg16 import (VGG16, preprocess_input, decode_predictions)
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential,load_model, model_from_json
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2,os,glob
import h5py

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path) #modify
    #img = image.load_img(img_path,grayscale=True) #modify
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    #x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    #print('2',layer_dict)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        #new_model = VGG16(weights='imagenet')
        new_model = model
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)
    
    nb_classes = len(predictions[0]) #class number #modify
    
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer, output_shape = target_category_loss_output_shape))
    loss = K.sum(model.layers[-1].output)
    conv_output = [l for l in model.layers[0].layers if l.name == layer_name][0].output
    #conv_output = [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    #print(grads)
    gradient_function = K.function([model.layers[0].input, K.learning_phase()], [conv_output, grads])

    output, grads_val = gradient_function([image,0])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))#GAP
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    
    #image.shape=image size
    #cam = cv2.resize(cam, (image.shape[2],image.shape[1]), interpolation=cv2.INTER_NEAREST) #tile version
    cam = cv2.resize(cam, (image.shape[2],image.shape[1])) #liner version #modify

#0618        
    cam = np.maximum(cam, max(0,cam.mean()-2*cam.std()))#Relu
    print('max:{}, mean:{}, std:{}'.format(cam.max(),cam.mean(),cam.std()))
    heatmap = (cam-np.min(cam)) / (np.max(cam)-np.min(cam))

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    
    print('max:{}, mean:{}, std:{},min:{}'.format(heatmap.max(),heatmap.mean(),heatmap.std(),heatmap.min()))
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)    
    cam = np.float32(cam)*1. + np.float32(image)*255*0.01
    cam = 255 * cam / np.max(cam)
    #print('add',cam[::3,::3,1]//1)
    return np.uint8(cam), heatmap

#modify,model,weight,target_conv
dataset_path="BVH_20180710_02_mix/"
model_path = dataset_path+'weights/bvh_model.json'
model_weight_path = dataset_path+'weights/BVH_weights_1.h5'
target_conv = "conv2_2"
print('model_path=', model_path)
print('model_weight_path=', model_weight_path)
print('target_conv=', target_conv)
    
#model,weight download
json_string = open(model_path).read()#modify
model = model_from_json(json_string)#modify
model.load_weights(model_weight_path)#modify
#model = load_model(model_weight_path)#modify
model.summary()


#image_path = 'BVH_all/miss/160-AI_stain-Real_BVH.bmp'
image_path = dataset_path+'miss'
image_file = [x  for x in glob.glob(image_path + "/*.bmp")]
image_save_path= dataset_path+'miss'
print('image_filenum',len(image_file))

for i in image_file:
    image_path1,image_file2=os.path.split(i)
    image_file2,image_file3=os.path.splitext(image_file2)
    print(image_file2)
    
    #image download
    #preprocessed_input = load_image(sys.argv[1])
    preprocessed_input = load_image(i)
    print('image_path=',i)
    #prediction,image
    predictions = model.predict(preprocessed_input)
    #top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:',predictions)
    #print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    #cam make

    predicted_class = np.argmax(predictions)
    print('target_class=',predicted_class)
    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, target_conv)
#file_name
    cam_name =os.path.join(image_save_path, image_file2 +str(predictions[0][predicted_class])+"_gcam2_2.jpg")
    print('cam_name=',cam_name)
    cv2.imwrite(cam_name, cam)
    
    #guided-grad-cam
    #register_gradient()
    #guided_model = modify_backprop(model, 'GuidedBackProp')
    #saliency_fn = compile_saliency_function(guided_model,target_conv)
    #saliency = saliency_fn([preprocessed_input, 0])
    #gradcam = saliency[0] * heatmap[..., np.newaxis]
    
    #guided_gradcam =os.path.join(image_save_path, image_file2+"_guided_gradcam.jpg")
    #print('guided_gradcam=', guided_gradcam)
    #cv2.imwrite(guided_gradcam, deprocess_image(gradcam))
    print(' ')