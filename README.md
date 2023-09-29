# Face Recognition using DeepFace


# Description 
In this face recognition project ,  I have used `DeepFace` framework to recognize people's name .


# How to install 
```
pip install -r requirements.txt 
```


# How to run 

You only need to Run `Deepface_recognition.py` . At the end of the code , in predict stage you should give an image to the model , to predict it's label .  
therefore set `img_path` to your test image path :
``` 
embedding_obj_features = DeepFace.represent(img_path="image.jpg" , model_name="ArcFace" , enforce_detection=False)

```


# RESULTS 
Here is our loss and accuracy results :

|| Accuracy  | Loss |
| ------------ | ------------- | ------------- |
train  | 0.96  | 0.11 |
test   | 0.76  | 1.96 |


<p float="center">
    <img src  = "assets\value4.JPG" width=650 /> 
</p>

<p float="center">
    <img src  = "assets\value4.JPG" width=600 /> 
</p>


# Predict Result 
for example :
```
embedding_obj_features = DeepFace.represent(img_path="Leyla-Hatami-07_01.jpg" , model_name="ArcFace" , enforce_detection=False)
```
<p float="center">
    <img src  = "assets\predict_res.JPG" width=600 /> 
</p>
