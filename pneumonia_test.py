from flask import Flask,render_template,request

from keras.utils import load_img
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np


app = Flask(__name__)
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        name=request.form['name']
        img=request.files['img']
        img.save('static/xray_img.jpg')
        model=load_model('chest_xray.h5')

        img=load_img('static/xray_img.jpg',target_size=(224,224))
        x=img_to_array(img)
        x=np.expand_dims(x, axis=0)
        img_data=preprocess_input(x)
        classes=model.predict(img_data)

        result=int(classes[0][0])

        if result==0:
            msg= f"{name} is Affected By PNEUMONIA"
        else:
            msg=f"{name} is not Affected"

        return render_template('pneumonia_detection.html',res=msg)
    else:
        return render_template('pneumonia_detection.html')

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")