import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data
from data import get_kitti_train_test_data
from callbacks import get_kitti_callbacks

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.utils import plot_model  # multi_gpu_model

from unet import *

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='kitti', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='depth_kitti_unet', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1: 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

# Create the model
myunet = myUnet(352,1216)
model = myunet.get_unet()

# Data loaders
if args.data == 'kitti': train_generator, test_generator = get_kitti_train_test_data( args.bs )

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if True: #False
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+'train.py', 'w') as training_script: training_script.write(training_script_content)

    # Generate model plot
    plot_model(model, to_file=runPath+'/model_plot.jpg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Multi-gpu setup:
basemodel = model  # https://discuss.tensorflow.org/t/multi-gpu-model/11778
# if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')

model.compile(loss='huber_loss', optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'kitti': callbacks = get_kitti_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)
if args.data == 'unreal': callbacks = get_kitti_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)

save_name = '/kitti' + '_ep_' + str(args.epochs) + '_bs_' + str(args.bs) + '_lr_' + str(args.lr) + '.h5'
print('Output model name: ',runPath + save_name)

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

# Save the final trained model
model.save(runPath + save_name)