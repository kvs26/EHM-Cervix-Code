#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, array_to_img, load_img
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# DIfferent Transfer Learning Models :- 

from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception

from keras.applications.resnet import ResNet50
from keras.applications.resnet_v2 import ResNet152V2

from keras.applications.mobilenet_v2 import MobileNetV2

from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet201

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.nasnet import NASNetLarge

from keras.applications.efficientnet import EfficientNetB5
from keras.applications.efficientnet_v2 import EfficientNetV2L


# In[ ]:


height = 512
width = 512
channels = 3

# path1 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EA\\"
# path2 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EH_Simple\\"
# path3 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EP\\"
# path4 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\NE_Follicular\\"
# path5 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\NE_Luteal\\"

path1 = "/home/201112046/perl5/Dataset_ED/EA/"
path2 = "/home/201112046/perl5/Dataset_ED/EH_Simple/"
path3 = "/home/201112046/perl5/Dataset_ED/EP/"
path4 = "/home/201112046/perl5/Dataset_ED/NE_Follicular/"
path5 = "/home/201112046/perl5/Dataset_ED/NE_Luteal/"

path = [path1, path2, path3, path4, path5]

def load_images_by_category(p):
    cat = []
    Class=os.listdir(p)
    for a in Class:
            try:
                image=cv2.imread(p+a)
                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                cat.append(np.array(size_image))
            except AttributeError:
                continue
    return cat

# Converting data to ndarray - 

cat1 = np.array(load_images_by_category(path1))
cat2 = np.array(load_images_by_category(path2))
cat3 = np.array(load_images_by_category(path3))
cat4 = np.array(load_images_by_category(path4))
cat5 = np.array(load_images_by_category(path5))


# In[ ]:


# Image Augmentation using Keras-ImageDataGenerator --> Rotation, horizontal & vertical flip....

def image_augmentation_by_category(folder, data):

    datagen = ImageDataGenerator(
            rotation_range=45,     #Random rotation between 0 and 45
            width_shift_range=0.2,   # % shift
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip = True,
            fill_mode='reflect', cval=125)    #Also try nearest, constant, reflect, wrap

    final_path = os.path.join("augmented_endometrial", folder)
    i = 1
    for batch in datagen.flow(data, 
                             batch_size=32,  
                             save_to_dir=final_path, 
                             save_prefix='aug', 
                             save_format='png'):
        i += 1
        if i > 64:
            break 


# In[ ]:


image_augmentation_by_category("EA", cat1)
image_augmentation_by_category("EH_Simple", cat2)
image_augmentation_by_category("EP", cat3)
image_augmentation_by_category("NE_Follicular", cat4)
image_augmentation_by_category("NE_Luteal", cat5)


# In[17]:


# height = 512
# width = 512
# channels = 3

# path1 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EA\\"
# path2 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EH_Simple\\"
# path3 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\EP\\"
# path4 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\NE_Follicular\\"
# path5 = r"C:\Users\vardh\OneDrive\Desktop\Major Project\Dataset_ED\NE_Luteal\\"

path1 = "/home/201112046/perl5/augmented_endometrial/EA/"
path2 = "/home/201112046/perl5/augmented_endometrial/EH_Simple/"
path3 = "/home/201112046/perl5/augmented_endometrial/EP/"
path4 = "/home/201112046/perl5/augmented_endometrial/NE_Follicular/"
path5 = "/home/201112046/perl5/augmented_endometrial/NE_Luteal/"

path = [path1, path2, path3, path4, path5]

# Finally collecting all the images from each category and creating labels....

data = []
labels = []
i = 0

for p in path:
    Class=os.listdir(p)
    for a in Class:
        try:
            image=cv2.imread(p+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            continue
    i+=1
    
# Converting data to ndarray - 
labels = np.array(labels)
data = np.array(data)

print(data.shape)
print(labels.shape)
print(labels)


# In[18]:


# Randomize the order of the input images

s = np.arange(data.shape[0])
np.random.seed(43)
np.random.shuffle(s)
data = data[s]
labels = labels[s]


# In[19]:


categories = ["EA", "EH_Simple", "EP", "NE_Follicular", "NE_Luteal"]

def show_the_image(i):
    plt.imshow(data[i])
    plt.xlabel(categories[labels[i]])


# In[20]:


show_the_image(34)


# In[21]:


# Loading ResNet50 with imagenet weights, include_top means that we loading model without last fully connected layers

model_resnet50          = ResNet50(weights = 'imagenet', include_top = False)
model_vgg19             = VGG19(weights = 'imagenet', include_top = False)
model_xception          = Xception(weights = 'imagenet', include_top = False)
model_mobilenetv2       = MobileNetV2(weights = 'imagenet', include_top = False)
model_resnet152v2       = ResNet152V2(weights = 'imagenet', include_top = False)
model_inceptionresnetv2 = InceptionResNetV2(weights = 'imagenet', include_top = False)
model_densenet201       = DenseNet201(weights = 'imagenet', include_top = False)
model_efficientnetv2l   = EfficientNetV2L(weights = 'imagenet', include_top = False)


# In[22]:


# Feature-Fusion-Function:-

def extract_features(f1,f2,f3,labels):
    extracted_features = []
    for f_r, f_d, f_e, label in zip(f1, f2, f3, labels):
        f_r = f_r.reshape(f_r.shape[0] * f_r.shape[1] * f_r.shape[2])
        f_d = f_d.reshape(f_d.shape[0] * f_d.shape[1] * f_d.shape[2])
        f_e = f_e.reshape(f_e.shape[0] * f_e.shape[1] * f_e.shape[2])

        f_r = np.append(f_r, f_d)
        f_r = np.append(f_r, f_e)
        f_r = np.append(f_r, label)
        extracted_features.append(f_r)
        
    return extracted_features


# In[23]:


# Hybrid Model:-

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def hybrid_model(model1, model2, model3):
    
    # Generating Features :-
    features_model1 = model1.predict(data, batch_size=32)
    features_model2 = model2.predict(data, batch_size=32)
    features_model3 = model3.predict(data, batch_size=32)
    
    print(features_model1.shape)
    print(features_model2.shape)
    print(features_model3.shape)
    
    # Combining Features :-
    extracted_features = np.array(extract_features(features_model1,  features_model2,  features_model3, labels))
    print(extracted_features.shape)
    
    # Normalization :-
    X = extracted_features[:, 0:-1]
    y = extracted_features[:, -1]
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    
    # PCA :-
    
    pca = PCA(n_components = 0.99)
    pca.fit(scaled_X)
    principal_X = pca.transform(scaled_X)
    # Check the values of eigen vectors prodeced by principal components
    print(len(pca.components_))
    # check how much variance is explained by each principal component
    print(pca.explained_variance_ratio_)
    
    # Train-Test-Split :-
    
    X_train, X_test, y_train, y_test = train_test_split(principal_X, y, test_size = 0.3, random_state = 10)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    # SVM for Classification :-

    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train, y_train)
    y_model = svm.predict(X_test)
    print(y_model[0:10])
    print(y_test[0:10])
    
    return (y_model, y_test)


# In[24]:


def ensemble():
    
    hybrid1 = hybrid_model(model_efficientnetv2l, model_densenet201, model_inceptionresnetv2)
    hybrid2 = hybrid_model(model_vgg19, model_efficientnetv2l, model_resnet50)
    hybrid3 = hybrid_model(model_resnet152v2, model_efficientnetv2l, model_densenet201)
    hybrid4 = hybrid_model(model_mobilenetv2, model_efficientnetv2l, model_densenet201)
    hybrid5 = hybrid_model(model_vgg19, model_mobilenetv2, model_densenet201)
    hybrid6 = hybrid_model(model_resnet152v2, model_resnet50, model_densenet201)
    hybrid7 = hybrid_model(model_resnet50, model_efficientnetv2l, model_xception)
    
    y_final = []
    y_test = hybrid1[1]
    
   
    for y1,y2,y3,y4,y5,y6,y7 in zip(hybrid1[0], hybrid2[0], hybrid3[0], hybrid4[0], hybrid5[0], hybrid6[0], hybrid7[0]):
        cate_count = [0, 0, 0, 0, 0]
            
        cate_count[int(y1)] += 1
        cate_count[int(y2)] += 1
        cate_count[int(y3)] += 1
        cate_count[int(y4)] += 1
        cate_count[int(y5)] += 1
        cate_count[int(y6)] += 1
        cate_count[int(y7)] += 1
        
        (c, r) = (0,0)
        for i in range(4):
            if(cate_count[i] > c):
                c = cate_count[i]
                res = i
                
        y_final.append(res)

            
    return (y_final, y_test)


# In[1]:


# Final output vector :-
call_ensemble = ensemble()
y_final = np.array(call_ensemble[0])
y_test = call_ensemble[1]


# In[ ]:


from sklearn import metrics

acc_svm = metrics.accuracy_score(y_test, y_final)
print("Accuracy of SVM: ", acc_svm*100)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

print("Classification Report: \n", classification_report(y_test, y_final))


# In[ ]:




