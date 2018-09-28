import os
import tensorflow as tf
import keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config =tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
K.set_session(sess)
import numpy as np
np.random.seed(777)
from keras.models import *
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from model import *
from data_0516 import *
import cv2
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image

#def train(dataset_path):
def train(dataset_path):
    model = get_model()
#   model = load_model(model_path)
    
    checkpoint_path="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    print(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0,\
                                 save_best_only=True, save_weights_only=True, mode='auto')     
    checkpoint_path2="weights/BVH_weights.h5"
    checkpoint2 = ModelCheckpoint(checkpoint_path2, monitor='val_loss', verbose=0,\
                                  save_best_only=True, save_weights_only=True, mode='auto')    
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    csv_logger = CSVLogger('weights/training.log')



    #X, y,val_X, val_y, test_X, test_y  = load_inria_person(dataset_path)
    X_all, y_all,y_label,idx  = load_middle_for_kfold(dataset_path)
    print(len(X_all.shape),X_all.shape)
    # print("Training..")
    val_num=len(set(idx))
    score_all=np.zeros((val_num,2))
    predict_classes_all=np.empty(0)
    true_classes_all=np.empty(0)
    
    for i in range(val_num):
        val_X = X_all[np.where(idx== ((i+1) %val_num) )]
        val_y = y_all[np.where(idx== ((i+1) %val_num) )]
        
        X_all2 = X_all[np.where(idx!=((i+1) %val_num) )]
        y_all2 = y_all[np.where(idx!=((i+1) %val_num) )]
        idx2 = idx[np.where(idx!=((i+1) %val_num))]
        
        test_X = X_all2[np.where(idx2==i)]
        test_y = y_all2[np.where(idx2==i)]
        test_y_label=y_label[np.where(idx2==i)]
        print(test_X.shape, test_y_label.shape)
        X = X_all2[np.where(idx2!=i)]
        y = y_all2[np.where(idx2!=i)]
        
        model.load_weights('weights/BVH_weights_org3.h5')
        history = model.fit(X, y, epochs=20, batch_size=16, validation_data=(val_X,val_y),\
                        verbose=1, callbacks=[csv_logger,checkpoint2])

        model_json_str = model.to_json()
        open('weights/bvh_model.json', 'w').write(model_json_str)
        #model.save_weights('bvh_weights.h5')
    
        model.load_weights('weights/BVH_weights.h5')
        score = model.evaluate(test_X, test_y,verbose=0)
        print(score)
        score_all[i]=score[0],score[1]
       
        predictions = model.predict(test_X)
        predict_classes = model.predict_classes(test_X, batch_size=32,verbose=0)
        true_classes = np.argmax(test_y,1)
        print(confusion_matrix(true_classes, predict_classes))
        predict_classes_all=np.append(predict_classes_all,predict_classes)
        true_classes_all=np.append(true_classes_all,true_classes)
        
        #miss data output
        catego = [ "stain","BVH"]
        #pre = model.predict(test_X)
        for j,v in enumerate(predict_classes):
            #pre_ans = v.argmax()
            ans = test_y[j].argmax()
            dat = test_X[j]
            if ans == v: continue
            fname = dataset_path+"/miss/"+"AI_"+catego[v]+"_Real_"+catego[ans]+\
                                str(predictions[j][v])+'_'+str(j)+".bmp"
            dat *= 255
            cv2.imwrite(fname,dat)
    
    print(confusion_matrix(true_classes_all, predict_classes_all))    
    print('Test loss:', score_all[:,0].mean())
    print('Test accuracy:', score_all[:,1].mean())
            
if __name__ == '__main__':
    dataset_path="BVH_all"
    train(dataset_path)
