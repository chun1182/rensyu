import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

def load_inria_person(path,contrast=False,rotation=False):
    
    pos_path = os.path.join(path, "train/OK_all/*")#train/big_class/middle
    neg_path = os.path.join(path, "train/NG_all/*")
    print(pos_path)
    pos_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(pos_path + "/*.bmp")]
    neg_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(neg_path + "/*.bmp")]
#    pos_images = [cv2.imread(x) for x in glob.glob(pos_path + "/*.jpg")]
#    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images] #for th
    print(len(pos_images),len(neg_images))

    val_pos_path = os.path.join(path, "val/OK_all/*")
    val_neg_path = os.path.join(path, "val/NG_all/*")
    val_pos_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(val_pos_path + "/*.bmp")]
    val_neg_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(val_neg_path + "/*.bmp")]
    print(len(val_pos_images),len(val_neg_images))

    test_pos_path = os.path.join(path, "test/OK_all/*")
    test_neg_path = os.path.join(path, "test/NG_all/*")
    test_pos_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(test_pos_path + "/*.bmp")]
    test_pos_name = [x for x in glob.glob(test_pos_path + "/*.bmp")]
    test_neg_images = [cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(test_neg_path + "/*.bmp")]
    test_neg_name = [x for x in glob.glob(test_neg_path + "/*.bmp")]
    print(len(test_pos_images),len(test_neg_images))

    y = [1] * len(pos_images) + [0] * len(neg_images)
    val_y = [1] * len(val_pos_images) + [0] * len(val_neg_images)
    test_y = [1] * len(test_pos_images) + [0] * len(test_neg_images)
    
    y = to_categorical(y, 2)
    val_y = to_categorical(val_y, 2)
    test_y = to_categorical(test_y, 2)

    
    if contrast:
        print('use contrast ajust')
        #expand contrast 0-255 every image
        X = np.float32([ (i-i.min()) / (i.max()-i.min())  for i in pos_images + neg_images])
        val_X = np.float32([ (i-i.min()) / (i.max()-i.min())  for i in val_pos_images + val_neg_images])
        test_X = np.float32([ (i-i.min()) / (i.max()-i.min())  for i in test_pos_images + test_neg_images])
    else:
        X = np.float32(pos_images + neg_images)/255.
        val_X = np.float32(val_pos_images + val_neg_images)/255.
        test_X = np.float32(test_pos_images + test_neg_images)/255.
       
    '''#-mean
    X=X-X.mean()
    val_X=val_X-X.mean()
    test_X=test_X-X.mean()
    
    datagen = ImageDataGenerator(zca_whitening=True)
    datagen.fit(X)
    g=datagen.flow(X,y, len(X),shuffle=False)
    g2=datagen.flow(test_X,test_y, len(X),shuffle=False)
    X,y=g.next()
    test_X,test_y=g2.next()'''

    #rotation train data
    if rotation:
        print('use rotation')
        #rotation can be made by flip and transpose
        X90=X.transpose(0,2,1,3)[:,:,::-1,:]
        X180=X[:,::-1,:,:][:,:,::-1,:]
        X270=X.transpose(0,2,1,3)[:,::-1,:,:]
        X=np.append(X,X90,axis=0)
        X=np.append(X,X180,axis=0)
        X=np.append(X,X270,axis=0)
        y=np.tile(y,(4,1))
    
    print('x:',X.shape,'y:',y.shape)
#    sys.exit()
    test_name = test_pos_name+test_neg_name

    return X, y, val_X, val_y, test_X, test_y,test_name
#    return X, y, val_X, val_y

def load_middle_for_kfold(path):
    neg_images,neg_y=[],[]
    label=dict()
    neg_middle=os.listdir(path+'/train/NG_all')
    for i,j in enumerate(neg_middle):
        middle_file=[cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(path+'/train/NG_all/'+j+ "/*.bmp")]
        neg_images+=middle_file
        neg_y+=[2*i for _ in range(len(middle_file))]
        label[2*i]=j
    
    pos_images,pos_y=[],[]
    pos_middle=os.listdir(path+'/train/OK_all')
    for i,j in enumerate(pos_middle):
        middle_file=[cv2.resize(cv2.imread(x), (30, 30)) for x in glob.glob(path+'/train/OK_all/'+j+ "/*.bmp")]
        pos_images+=middle_file
        pos_y+=[2*i+1 for _ in range(len(middle_file))]
        label[2*i+1]=j
    
    X=np.float32(pos_images + neg_images)/255.
    y_label=pos_y+neg_y
    y_label=np.array(y_label)
    val_idx=[]
    idx=np.zeros(len(y_label))
    
    for train_idx, test_idx in StratifiedKFold(n_splits=6,random_state=1,shuffle=True).split(y_label,y_label):
        val_idx.append([train_idx,test_idx])
    for k,(i,j) in enumerate(val_idx):
        idx[j]=k

    y=y_label%2
    y = to_categorical(y, 2)
    y_label=np.array([label[i] for i in y_label])

    X = np.float32(pos_images + neg_images)/255.
    return X, y,y_label, idx