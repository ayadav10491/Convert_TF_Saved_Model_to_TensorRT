"""
The implementation of some utils.

@Author: Akash Yadav
@Github: https://github.com/ayadav10491
@Project: https://github.com/ayadav10491/Convert_TF_Saved_Model_to_TensorRT

"""

#Based on experimental tensorflow to tensorrt converter on tf-nightly=2.3 (check versions in requirement.txt)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils.losses import categorical_crossentropy_with_logits
import keras.backend.tensorflow_backend as K
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from utils.helpers import get_dataset_info
from utils.utils import load_image
from keras_applications import imagenet_utils
import numpy as np
import cv2

backend = tf.compat.v1.keras.backend

parser = argparse.ArgumentParser()


parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='CamVid')
parser.add_argument('--saved_model_folder', help='The path of the dataset.', type=str, default='tf_saved_model')
parser.add_argument('--save_trt_model', help='The path of the dataset.', type=str, default='TRT_Model')

args = parser.parse_args()


params = tf.experimental.tensorrt.ConversionParams(
    precision_mode='FP16')


converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir=args.saved_model_folder, conversion_params=params)

train_image_names, _, _, _, _, _ = get_dataset_info(
              args.dataset)
width=256
height=256

### Creating representation function for dataset to facililitate TRT engine building
### if do not wish create TRT engine comment out converter.build(input_fn=representative_data_gen)
### and then representative data won't be required as well.
for i, name in enumerate(train_image_names):

            image=load_image(name)
            image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last',
                                                    mode='torch')
            image=cv2.resize(image,(width,height),interpolation=cv2.INTERSECT_NONE)
            image = np.expand_dims(image, axis=0)
            dataset_ = tf.data.Dataset.from_tensor_slices((image)).batch(1)

def representative_data_gen():
            for input_value in dataset_.take(10):
                yield [input_value]



#converter.convert(calibration_input_fn=representative_data_gen) INT8 Conversion Still in development
converter.convert()
converter.build(input_fn=representative_data_gen)

converter.save(args.save_trt_model)

