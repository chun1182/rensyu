import os,csv,cv2
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
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau
from model_0706 import get_model, get_random_eraser, make_intermediate_layer_model
from data_0706 import load_inria_person
#import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
#from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
#def train(dataset_path):
def train(dataset_path,cnt):
    batch_size=32
    epochs=50
    data_argumentation=2 #argumentation=0, not use arg=1, no learn=2
    model = get_model()
#   model = load_model(model_path)
    
    time1 = time.time()
    X, y,val_X, val_y, test_X, test_y,test_name  = load_inria_person(dataset_path)
    print(len(X),X.shape)
    time2 = time.time()
    

    # print("Training..")
    '''#setting callback,checkpoint,logger,lr_reducer'''
    checkpoint_path=dataset_path+"/weights/weights.{epoch:02d}-{val_loss:.3f}.h5"
    print(checkpoint_path)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0,\
                                 save_best_only=True, save_weights_only=True, mode='auto')  
   
    checkpoint_path2=dataset_path+"/weights/BVH_weights_"+str(cnt)+".h5"
    checkpoint2 = ModelCheckpoint(checkpoint_path2, monitor='val_acc', verbose=0,\
                                  save_best_only=True, save_weights_only=True, mode='auto')
    
    earlystop = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    #csv logger ,added the file
    csv_logger = CSVLogger(dataset_path+'/weights/training.csv',append=True)
    
    #reduce leaning rate when not update patience num
    lr_reducer = ReduceLROnPlateau(factor=0.1**0.5, cooldown=0, patience=20,
                                   verbose=1, min_lr=0.5e-6)
    call_backs=[checkpoint ,csv_logger,checkpoint2,lr_reducer]

    if data_argumentation==0:
        print('using data argumentation')
        #ON: shift, horizontan and vertical flip, random erasing when 50% cover 10~30% color 0.3~0.9
        datagen = ImageDataGenerator(width_shift_range=0.05,height_shift_range=0.05,
                                           horizontal_flip=True,vertical_flip=True,
                   preprocessing_function=get_random_eraser(s_l=0.1, s_h=0.3, v_l=0.3, v_h=0.9))
        datagen.fit(X)
        model.fit_generator(datagen.flow(X, y, batch_size=batch_size),
                        steps_per_epoch=X.shape[0] // batch_size, epochs=epochs,
                        validation_data=(test_X,test_y),callbacks=call_backs)
    elif data_argumentation==1:
        print('not using data argumentation')
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                            validation_data=(val_X,val_y),verbose=1, callbacks=call_backs)

    model_json_str = model.to_json()
    open(dataset_path+'/weights/bvh_model.json', 'w').write(model_json_str)
    
    #use max val_acc from all of epoch 
    #model.load_weights(checkpoint_path2)
    model.load_weights(dataset_path+'/weights/BVH_weights_4.h5')
    score = model.evaluate(test_X, test_y,verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    predictions = model.predict(test_X)
    print('predictions,shape',predictions.shape)
    predict_classes = model.predict_classes(test_X, batch_size=32,verbose=0)
    true_classes = np.argmax(test_y,1)
    print(confusion_matrix(true_classes, predict_classes))
    
    f = open(dataset_path+'/miss/writeDataHRibutsu_'+str(cnt)+'.csv', 'w')
    dataWriter = csv.writer(f)
    rowlist = [['name','big','middle','AI','Real','judge']]
    
    #miss data output
    catego = [ "stain","BVH"]
    #pre = model.predict(test_X)
    
    '''miss data save / csv output
    for i,v in enumerate(predict_classes):
        #pre_ans = v.argmax()
        ans = test_y[i].argmax()
        dat = test_X[i]
        name=test_name[i].replace('\\','/').split('/')
        
        #for csv output
        if ans==v: rowlist.append([name[4],name[2],name[3],catego[v],catego[ans],'o'])
        else: rowlist.append([name[4],name[2],name[3],catego[v],catego[ans],'x'])
        
        #only miss judge
        if ans == v: continue
        fname = dataset_path+"/miss/"+str(i)+"-AI_"+catego[v]+"-Real_"+catego[ans]+\
                '_'+ str(predictions[i][v])+".bmp"
        dat *= 255
        cv2.imwrite(fname,dat)
    '''
    
    dataWriter.writerows(rowlist)
    f.close()
    
    time3 = time.time()
    interval1,interval2  = time2 - time1,time3 - time2
    print(str(interval1) + "sec",str(interval2) + "sec")
        
if __name__ == '__main__':
    dataset_path="BVH_20180710_02_mix"
    #dataset_path="BVH_all_old"
    for cnt in range(1):
        train(dataset_path,cnt+1)
