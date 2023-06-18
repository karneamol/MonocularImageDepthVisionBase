import os
import glob
import argparse
import matplotlib
import numpy
from PIL import Image

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images, save_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.jpg', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='examples/*.jpg', type=str, help='Output filename or folder.')
args = parser.parse_args()

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
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
#viz = display_images(outputs.copy(), inputs.copy())
#viz = numpy.reshape(outputs[0],(352,1216))
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.savefig(args.model.replace('h5','png'))
#plt.show()

viz = save_images(args.model.replace('h5','png'), outputs.copy(), inputs.copy())

# Save all outputs
ctr = 0
for o, f in zip(outputs,glob.glob(args.input)):
    viz = numpy.reshape(o,(352,1216)) * 255
    image = Image.fromarray(viz.astype('uint8'))
    image.save(outputDir +'/' + f[5:])
    ctr = ctr + 1