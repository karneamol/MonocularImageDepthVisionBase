import os
import glob
import argparse
import matplotlib
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sys import getsizeof
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
#from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, save_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.jpg', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='examples/*.jpg', type=str, help='Output filename or folder.')
args = parser.parse_args()

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

def standardize(image_data):
        image_data -= np.mean(image_data, axis=0)
        image_data /= np.std(image_data, axis=0)
        return image_data
# Custom object needed for inference and training
#custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Output location
outputDir = args.output

# Load model into GPU / CPU
#model = load_model(args.model, custom_objects=custom_objects, compile=False)
model = load_model(args.model, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images
inputs = load_images( glob.glob(args.input) )
print('\nLoaded ({0}) images of size {1}.  and of datatype {2}'.format(inputs.shape[0], inputs.shape[1:], inputs.dtype))

# Compute results ###########Commented
# outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
#viz = display_images(outputs.copy(), inputs.copy())
#viz = numpy.reshape(outputs[0],(352,1216))
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.savefig(args.model.replace('h5','png'))
#plt.show()

########### Find Keras Model Size
keras_model_size = get_file_size(args.model)
keras_model_size_KB = convert_bytes(keras_model_size, "KB")
print("Original Keras model size in KB",keras_model_size_KB)
########### Convert Keras Model to TFLite model
TF_LITE_MODEL_FILE_NAME = "tf_lite_kitti-depth-unet.tflite"
converter = tf.lite.TFLiteConverter.from_keras_model( model ) 
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
file = open( TF_LITE_MODEL_FILE_NAME , 'wb' ) 
file.write( tflite_model )
########### Find TF Lite Model Size
tflite_model_size = get_file_size(args.model)
tflite_model_size_KB = convert_bytes(keras_model_size, "KB")
print("Converted TFLite model size in KB",tflite_model_size_KB)

########### Test converted tensorflow lite model
########### Find input details and output details for tflite model
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

########### In case input and output parameters of models are to be changed. Use below code.
interpreter.resize_tensor_input(input_details[0]['index'], (inputs.shape[0],352,1216,3))
interpreter.resize_tensor_input(output_details[0]['index'], (inputs.shape[0],352,1216,1))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])

########### Convert input images datatypes
test_images = inputs
print("Input test images are of datatype", test_images.dtype)
test_imgs_numpy = np.array(test_images, dtype=np.float32)
print("Input test images are converted to datatype", test_imgs_numpy.dtype)

########### Predict depth from images
interpreter.set_tensor(input_details[0]['index'], test_imgs_numpy)
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", tflite_model_predictions.shape)
prediction_classes = np.argmax(tflite_model_predictions, axis=1)
outputs = tflite_model_predictions

print("printing output array 1", outputs[0])

viz = save_images(args.model.replace('.h5','tflite.png'), outputs.copy(), inputs.copy())

# Save all outputs
ctr = 0
for o, f in zip(outputs,glob.glob(args.input)):
    viz = np.reshape(o,(352,1216)) * 255
    image = Image.fromarray(viz.astype('uint8'))
    #Create new output directory for tflite model outputs
#    os.makedirs('/content/depth-kitti-unet/output2', exist_ok = True)
    image.save(outputDir +'/' + f[5:])
    ctr = ctr + 1
