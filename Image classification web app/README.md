
# 
<div align="center">
  <strong><h1>👉 <a href="https://github.com/YounesseELH/Machine-Learning-Models-using-Flask/tree/main/Image%20classification%20web%20app">Image classification web app</a> 👈 </h1></strong>
</div>

## 📽️ Simulation 
![1c0b8b9a-b722-4b7d-b308-3b84540e1f3f](https://user-images.githubusercontent.com/96134357/167468879-c35b69f6-72e7-4f20-b375-9b35912d6731.gif)

## 🛠 Code : 

```python
from pyexpat import model
from flask import Flask,render_template,request

from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = VGG16()

@app.route('/',methods=['GET'])
def mGet():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def prediction():
    imageFile = request.files['image_file']
    imagePath = "./images/"+imageFile.filename
    imageFile.save(imagePath)

    image = load_img(imagePath,target_size=(224,224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    classification = '%s (%.2f%%' %(label[1],label[2]*100)
    return render_template('index.html',pred = classification)


if __name__ == '__main__':
    app.run(port=3000,debug=True)
```
