import warnings
warnings.filterwarnings('ignore')


from tensorflow import keras


from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE = [224, 224]

train_path = "D:\\chest_xray\\chest_xray\\train"
valid_path = "D:\\chest_xray\\chest_xray\\test"


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



for layer in vgg.layers:
    layer.trainable = False



folders = glob('D:\\chest_xray\\chest_xray\\train\\*')
x = Flatten()(vgg.output)


prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()



model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[25]:


from keras.preprocessing.image import ImageDataGenerator


# In[26]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('D:\\chest_xray\\chest_xray\\train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('D:\\chest_xray\\chest_xray\\test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')


# In[27]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

model.save('chest_xray.h5')





