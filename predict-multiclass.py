#Usage: python predict-multiclass.py
#https://github.com/tatsuyah/CNN-Image-Classifier

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Cherry Rainer")
  elif answer == 1:
    print("Label: Cherry Wax Black")
  elif answer == 2:
    print("Label: Cherry Wax Red")

  return answer

Cherry_Rainier_t = 0
Cherry_Rainier_f = 0
Cherry_Wax_Black_t = 0
Cherry_Wax_Black_f = 0
Cherry_Wax_Red_t = 0
Cherry_Wax_Red_f = 0

for i, ret in enumerate(os.walk('./test-data/CherryRainier')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Cherry Raiiner")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      Cherry_Rainier_t += 1
    else:
      Cherry_Rainier_f += 1

for i, ret in enumerate(os.walk('./test-data/CherryWaxBlack')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Cherry Wax Black")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      Cherry_Wax_Black_t += 1
    else:
      Cherry_Wax_Black_f += 1

for i, ret in enumerate(os.walk('./test-data/CherryWaxRed')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Cherry Wax Red")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      print(ret[0] + '/' + filename)
      Cherry_Wax_Red_t += 1
    else:
      Cherry_Wax_Red_f += 1

"""
Check metrics
"""
print("True Cherry Rainier: ", Cherry_Rainier_t)
print("False Cherry Rainier: ", Cherry_Rainier_f)
print("True Cherry Wax Black: ", Cherry_Wax_Black_t)
print("False Cherry Wax Black: ", Cherry_Wax_Black_f)
print("True Cherry Wax Red: ", Cherry_Wax_Red_t)
print("False Cherry Wax Red: ", Cherry_Wax_Red_f)
