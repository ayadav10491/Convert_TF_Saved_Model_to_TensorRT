"""
The implementation of some utils.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/Convert_TF_Saved_Model_to_TensorRT

"""





import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
from utils.losses import categorical_crossentropy_with_logits
import keras.backend.tensorflow_backend as K
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from keras_applications import imagenet_utils
import numpy as np
import os

# backend = tf.compat.v1.keras.backend  For TensorFlow 1.X versions

parser = argparse.ArgumentParser()


parser.add_argument('--model_name', help='The path of the trained model.', type=str, default='./CamVid_UNet_based_on_MobileNetV2_crop_256_epochs_200_classes_32_batch_size_4_None.h5')
parser.add_argument('--save_model_folder', help='The path of the trained model.', type=str, default='tf_saved_model')

args = parser.parse_args()


# Clear any previous session.
tf.keras.backend.clear_session()


model_path = os.path.join('weights',(args.model_name))

if os.path.isfile(model_path):
    print('Model ', model_path, ' exists')

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

#Here the loaded model is the part of semantic segmentation training done on CamVid.
#Please change according to your custom objects and import the function before initialiting here ..
model = load_model(model_path, custom_objects={
        'categorical_crossentropy_with_logits': categorical_crossentropy_with_logits})

print('Model Import Successful....................')

print('Initiating Conversion....................')

export_path=args.save_model_folder

""" SAVE MODEL # For TensorFlow 1.X versions
with backend.get_session() as sess:
         tf.compat.v1.saved_model.simple_save(
              sess,
              export_path,
              inputs={'input_image': model.input},
              outputs={t.name: t for t in model.outputs}
        )
"""
# SAVE MODEL # For TensorFlow 2.x versions
tf.saved_model.save(model, export_path)
print('Conversion Successful.....................')
