# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:44:48 2018

@author: F0004562
"""
from keras.models import Sequential,load_model, model_from_json,Model
import numpy as np
import cv2,os,glob,sys
import h5py
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.spatial import distance
from sklearn import cluster

def make_intermediate_layer_model(model, layer_name='Dense_2'):
    intermediate_layer_model=Model(input=model.input, output=model.get_layer(layer_name).output)
    return intermediate_layer_model

def user_num_image(path, user=('train', 'test', 'val'), contrast=False):
    user=('train','test','val')
    test_pos_path =[os.path.join(path, i, "OK_all/*") for i in user]
    test_neg_path =[os.path.join(path, i, "NG_all/*") for i in user]
    
    max_img_num=500
    X, y=[], []
    for j,i in enumerate(test_pos_path+test_neg_path):
        n=len(glob.glob(i + "/*.bmp"))//max_img_num+1
        X += [ cv2.resize(cv2.imread(x, 1), (30,30)) for k,x in enumerate(glob.glob(i + "/*.bmp")) if (k+1)%n==0]
        y+=[j]*(len(glob.glob(i + "/*.bmp"))//n)
        print(i,len(glob.glob(i + "/*.bmp")),(len(glob.glob(i + "/*.bmp"))//n))
    
    if contrast:
        print('use contrast ajust')
        #expand contrast 0-255 every image
        X = np.float32([ (i-i.min()) / (i.max()-i.min())  for i in X])
    else:
        X = np.float32(X)/255.
   
    return X, y

def plt_tsne(X_tsne,y):
    plt.figure(figsize=(14,13))
    plt.xlim(X_tsne[:,0].min(), X_tsne[:,0].max()+1)
    plt.ylim(X_tsne[:,1].min(), X_tsne[:,1].max()+1)
    leny=len(set(y))
    for i in range(len(X)):
        plt.text(X_tsne[i,0], X_tsne[i,1],str(y[i]+1),color=plt.cm.bwr(y[i]// (leny/2) ), fontdict={'weight':'bold', 'size':9})
    #plt.xlabel('tsne feature 0')
    #plt.ylabel('tsne feature 1')
    plt.xticks([]), plt.yticks([])
    plt.savefig('plt_tsne.png', bbox_inches='tight')
    #plt.show()

def plot_tsne_img(X,X2,y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(14,13))
    ax = plt.subplot(111)
    leny=len(set(y))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]+1),\
                 color=plt.cm.bwr(y[i] // (leny/2)), fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        plt.gray()
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 1e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            #imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X2[i].reshape(30,30),cmap=plt.cm.gray_r),X[i])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(X2[i].reshape(30,30,3)),X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:  plt.title(title)
    plt.savefig('plt_tsne_img.png', bbox_inches='tight')

if __name__ == '__main__':
    #データセット指定
    dataset_path="BVH_20180710_02_mix/"
    #使用するモデル＆重み＆ターゲット層指定
    model_path = dataset_path+'weights/bvh_model.json'
    model_weight_path = dataset_path+'weights/BVH_weights_4.h5'
    target_conv = "Dense_2"
    print('model_path=', model_path)
    print('model_weight_path=', model_weight_path)
    print('target_conv=', target_conv)

    #サンプル取得    
    X, y = user_num_image(dataset_path)
    print(len(X),X.shape)
    
    #モデル＆重み取得
    json_string = open(model_path).read()#modify
    model = model_from_json(json_string)#modify
    model.load_weights(model_weight_path)#modify
    #model = load_model(model_weight_path)#modify
    model.summary()
        
    intermediate_layer_model = make_intermediate_layer_model(model, layer_name='Dense_2')
    intermediate_layer_predictions = intermediate_layer_model.predict(X)
    print(intermediate_layer_predictions.shape)
    
    tsne= TSNE(random_state=42,perplexity=30.0)
    X_predict_tsne= tsne.fit_transform(intermediate_layer_predictions)
    plt_tsne(X_predict_tsne,y)
    plot_tsne_img(X_predict_tsne,X,y)

    #X_tsne= tsne.fit_transform(X.reshape(len(X),-1))
    #plt_tsne(X_tsne,y)
    #plot_tsne_img(X_tsne,X,y)
