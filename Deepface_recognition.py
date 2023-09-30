import os
import numpy as np
import pandas as pd
import tensorflow as tf
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split

main_folder_path = r'face_dataset'

folders_name = []

for file_path in os.listdir(main_folder_path) :
   folders_name.append(file_path)

print(folders_name)

complete_path_each_folder = []
for i in range(len(folders_name)):
  path = main_folder_path +"/"+ folders_name[i]
  complete_path_each_folder.append(path)


print(complete_path_each_folder)

def generate_dataset(path):
    embedding_obj_features = DeepFace.represent(img_path= path , model_name="ArcFace" , enforce_detection=False)
    return embedding_obj_features


dicti = defaultdict(list)

for k in range(0,len(folders_name)) :

  folder_path = complete_path_each_folder[k]
  print(folder_path)
  pictures_name = []
  for file_path in os.listdir(folder_path) :
    pictures_name.append(file_path)
  complete_path = []
  for p in range(len(pictures_name)):
    path = folder_path +"/"+ pictures_name[p]
    complete_path.append(path)

  for j in range(len(pictures_name)) :
      face_feature_numbers = generate_dataset(complete_path[j])
      dicti['Label'].append( folders_name[k])
      dicti['File_name'].append(pictures_name[j] )

      for i in range(len(face_feature_numbers[0]["embedding"])) :
            dicti[f'Feature{i+1}'].append(face_feature_numbers[0]["embedding"][i])

  for j in range(len(pictures_name)) :
      dicti['facial_area_x'].append( face_feature_numbers[0]["facial_area"]['x'] )
      dicti['facial_area_y'].append( face_feature_numbers[0]["facial_area"]['y'] )
      dicti['facial_area_w'].append( face_feature_numbers[0]["facial_area"]['w'] )
      dicti['facial_area_h'].append( face_feature_numbers[0]["facial_area"]['h'] )

  print(k)

df = pd.DataFrame(dicti)
df.to_csv("dataset/face_dataset.csv")

labels_array = df["Label"].unique()

df['num_label'] = 0
for i in range(len(df)) :
  for j in range(len(labels_array)):
    if df['Label'][i] == labels_array[j]:
      df['num_label'][i] = j



X_train = np.array(df.iloc[: , 2:-5])
Y_train = np.array(df["num_label"])

x_train ,  x_test , y_train , y_test = train_test_split(X_train ,Y_train ,  test_size=0.2)
y_test = y_test.reshape(-1 , 1)
y_train = y_train.reshape(-1 , 1)
print(x_train.shape , y_train.shape , x_test.shape , y_test.shape )
x_train , x_test = x_train / 255.0 , x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512 ),
    tf.keras.layers.Dense(256 , activation= "relu" ),
    tf.keras.layers.Dense(128 , activation= "relu" ),
    tf.keras.layers.Dense(30  , activation= "softmax")
    ])


model.compile(optimizer="adamax" , loss=tf.keras.losses.sparse_categorical_crossentropy , metrics="accuracy")
mymodel = model.fit(x_train , y_train , epochs=600)
loss , accuracy = model.evaluate(x_test , y_test )
model.save("weights/deep_face_recog.h5")
print("TEST LOSS" , loss)
print("TEST ACCURACY" , accuracy)

plt.plot(mymodel.history["loss"] )
plt.plot( mymodel.history["accuracy"])
plt.xlabel("epoches")
plt.ylabel("loss")
plt.title("train loss & accuracy")
plt.legend(["loss" , "accuracy"])
plt.savefig("assets/train_result1.jpg")



""" PREDICT image label """

def generate_dataset():
    embedding_obj_features = DeepFace.represent(img_path="face_dataset/Leyla_Hatami/Leyla-Hatami-07_01.jpg" , model_name="ArcFace" , enforce_detection=False)
    return embedding_obj_features

face_feature_numbers = generate_dataset()

test_img =np.array( face_feature_numbers[0]["embedding"] )
test_img = test_img.reshape(1 ,-1)

pred = model.predict(test_img)
print(pred)

pred = np.argmax(pred)
print(pred)

print(labels_array[pred])