# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers.schedules import ExponentialDecay
from keras import callbacks
from matplotlib.dates import drange
from tensorflow.keras.optimizers import SGD, Adam
from keras.optimizers.schedules import CosineDecay, PiecewiseConstantDecay
#from keras.utils import np_utils
from keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
from datetime import datetime
import numpy as np
import os

def display_classification_report(classification_report, figure_path, figure_name, onscreen=True):
    f = open(os.path.join(figure_path, figure_name+'.txt'), 'w', encoding='utf-8')
    f.write(classification_report)
    f.close()

    if onscreen:
       print(classification_report)

def display_confusion_matrix(confusion_matrix, labels, figure_path,figure_name,figure_format,onscreen=True):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Greys)

    plt.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)

    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          print("Show confusion matrix on display")

          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)

def display_activations(input, label, activations, layer_names, figure_path, figure_name, figure_format, onscreen=True):
    fig = plt.figure(layout='constrained', figsize=(10, 8))
    subfigs = fig.subfigures(1, len(activations)+1) # layers + input , figsize=(row_size*2.5,len(model.layers)*1.5))

    subfigs[0].subplots(1, 1)
    subfigs[0].suptitle('Label: {}'.format(label))
    axs = subfigs[0].get_axes()
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].imshow(input, cmap='gray_r')

    for layer_index in range(0,len(activations)):
        print("layer:" +str(layer_index))
        print(activations[layer_index].shape[-1])
        subfigs[layer_index+1].suptitle(layer_names[layer_index])
        subfigs[layer_index+1].subplots(activations[layer_index].shape[-1],1)

    for layer_index in range(0,len(activations)):
        print(activations[layer_index].shape)
        #range(0,activations.shape[-1]):
        axs = subfigs[layer_index+1].get_axes()
        for plane_index in range(0,activations[layer_index].shape[-1]):
            plane = activations[layer_index][0,:, :, plane_index]
            axs[plane_index].set_xticks([])
            axs[plane_index].set_yticks([])
            axs[plane_index].imshow(plane, cmap='gray_r')

    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)



def display_weights_column(weights, layer_names,figure_path,figure_name,figure_format,onscreen=True):
    n_layers_with_weights = 0
    for layer_index in range(0, len(weights)):
        layer_weights = weights[layer_index]
        if len(layer_weights) > 0:
            n_layers_with_weights += 1

    fig = plt.figure(figsize=(30, 15), frameon=False)

    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
    subfigs = fig.subfigures(1, n_layers_with_weights)


    layer_index_with_weights = 0
    print("Number of layers: "+str(len(weights)))
    for layer_index in range(0, len(weights)):
        layer_weights = weights[layer_index]

        print("layer:" +str(layer_index))
        # only weights (0) no biases (1)
        if len(layer_weights) > 0 and len(layer_weights[0].shape) > 1:
            print(" weights shape ", layer_weights[0].shape)

            #subfigs[layer_index_with_weights].suptitle(layer_names[layer_index])
            #subfigs[layer_index_with_weights].tight_layout()
            # squeeze=False squeezing at all is done: the returned Axes object is always a 2D array containing
            axs = subfigs[layer_index_with_weights].subplots(layer_weights[0].shape[-2], layer_weights[0].shape[-1], squeeze=False)#, sharex=True, sharey=True)
            subfigs[layer_index_with_weights].subplots_adjust(wspace=0.1, hspace=0.1, top=1.0, bottom=0.0, left=0.0, right=1.0)
            print(axs.shape)
            for i in range(0, layer_weights[0].shape[-2]):
                for j in range(0, layer_weights[0].shape[-1]):
                    w = layer_weights[0]
                    axs[i,j].imshow(w[:,:,i,j], cmap='gray_r', interpolation='nearest')
                    axs[i,j].axis("off")

            layer_index_with_weights += 1
    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)

    if onscreen:
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)


def display_loss_function(history,figure_path,figure_name,figure_format,onscreen=True):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    fig = plt.figure()
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig(os.path.join(figure_path,figure_name+'.'+figure_format), format=figure_format)
    if onscreen:
       print("Show loss on display")
       if matplotlib.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "png"]:
          plt.show()
       else:
          print("Non-interactive backend; figure saved but not shown.")
          plt.close(fig)
    else:
       plt.close(fig)

# loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# normalizing the data 
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
n_cnn1planes = 15
n_cnn1kernel = 3
n_poolsize = 1


# Stride defines the step size at which the filter moves across the input during convolution. 
# A larger stride results in a reduction of the spatial dimensions of the output feature map. 
# Stride can be adjusted to control the level of downsampling in the network.
# Stride is a critical parameter for controlling the spatial resolution of the feature maps and influencing the receptive field of the network.
n_strides = 1
n_dense = 100
dropout = 0.4

n_epochs=15
batch_size = 128

model_name = 'CNN_Handwritten_OCR_CNN'+str(n_cnn1planes)+'_KERNEL'+str(n_cnn1kernel)+'_Epochs' + str(n_epochs)+ '_Dropout_' + str(dropout)
#figure_format='svg'
figure_format='png'
figure_path='./'
log_path='./log'

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
cnn1 = Conv2D(n_cnn1planes, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu', input_shape=(28,28,1))
model.add(cnn1)
#cnn11 = Conv2D(n_cnn1planes, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
#model.add(cnn11)
model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))

#model.add(Dropout(dropout))

cnn2 = Conv2D(n_cnn1planes*2, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
model.add(cnn2)
#cnn22 = Conv2D(n_cnn1planes*2, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
#model.add(cnn22)
model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))

#model.add(Dropout(dropout))

cnn3 = Conv2D(n_cnn1planes*4, kernel_size=(n_cnn1kernel,n_cnn1kernel), strides=(n_strides,n_strides), padding='valid', activation='relu')
model.add(cnn3)
model.add(MaxPool2D(pool_size=(n_poolsize,n_poolsize)))

#model.add(Dropout(dropout))

# flatten output of conv
model.add(Flatten())


model.add(Dropout(dropout))

# hidden layer
model.add(Dense(n_dense, activation='relu'))
# output layer
model.add(Dense(n_classes, activation='softmax'))

# compiling the sequential model


# vary the constant learning rate
#learning_rate = 0.009

#model_name += '_Optimzer_SGD_LR_' + str(learning_rate)
#model_name += '_Optimzer_SGD'

# Update figure_path to point to this new folder


# OR use a learning rate scheduler that adapts the learning rate over the epochs of the training process
# https://keras.io/2.15/api/optimizers/learning_rate_schedules/


steps_per_epoch = len(X_train) // batch_size
total_decay_steps = n_epochs * steps_per_epoch
boundaries = [5 * steps_per_epoch, 10 * steps_per_epoch] # Drop after Epoch 5 and Epoch 10
#model_name += '_LearningRate_' + 'ExponentialDecay'
initial_lr = 0.1
values = [initial_lr, initial_lr * 0.1, initial_lr * 0.01]
#learning_rate = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=n_epochs, decay_rate=0.9)
#learning_rate=0.008
#momentum = 0.9
#learning_rate = CosineDecay(
#    initial_learning_rate=initial_lr,
#    decay_steps=total_decay_steps
#)
learning_rate = PiecewiseConstantDecay(
    boundaries=boundaries,
    values=values
)

#optimizer=SGD(learning_rate = learning_rate, momentum = momentum)
optimizer = SGD(learning_rate=learning_rate, momentum=0.0)
#optimizer=Adam(learning_rate = learning_rate)

# 2. Define Schedule


model_name += f"T3_StepDecay_Init{initial_lr}"
#model_name += f"T3_Cosine_Init{initial_lr}"
#model_name += '_Optimizer_SGD_ExpDecay_InitLR_' + str(initial_lr)
#model_name += '_Optimizer_SGD_Momentum_' + str(momentum) + '_CosDecay_'

# vary the constant learning rate
#learning_rate = 0.01
#optimizer=SGD(learning_rate=learning_rate)
if not os.path.exists(model_name):
    os.makedirs(model_name)
    print(f"Created new directory: {model_name}")
figure_path = model_name
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

layer_names = [layer.name for layer in model.layers[:8]]

weights = [layer.get_weights() for layer in model.layers[:4]]
figure_name=model_name + '_initial_weights'
display_weights_column(weights, layer_names, figure_path, figure_name, figure_format, False )

# training the model for n_epochs, use 10% of the training data as validation data
history = model.fit(X_train, Y_train, validation_split = 0.1, batch_size=128, epochs=n_epochs )

figure_name=model_name + '_loss'
display_loss_function(history,figure_path,figure_name,figure_format)

weights = [layer.get_weights() for layer in model.layers[:4]]
figure_name=model_name + '_weights' 
display_weights_column(weights, layer_names, figure_path, figure_name, figure_format, False )

X_test_images = X_test[:2]
for i in range(X_test_images.shape[0]):
    Y_test_pred = model.predict(np.expand_dims(X_test[i], axis=0))

    activation_model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
    activations = activation_model.predict(np.expand_dims(X_test[i], axis=0))

    figure_name=model_name + '_activations_' + 'test_image_' + str(i)
    display_activations(X_test[i],np.argmax(Y_test[i]),activations[:4], layer_names[:4], figure_path, figure_name, figure_format)

y_test_pred = model.predict(X_test)
cm = confusion_matrix(y_test, np.argmax(y_test_pred,axis=1), labels=range(0,n_classes))

figure_name=model_name + '_confusion_matrix'
display_confusion_matrix(cm, range(0,n_classes), figure_path,figure_name,figure_format)

figure_name=model_name + '_classification_report'
display_classification_report(classification_report(y_test, np.argmax(y_test_pred,axis=1), target_names=[str(c) for c in range(0,n_classes)], digits=4), figure_path, figure_name)

figure_name=model_name + '_model_summary'
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
model_summary = "\n".join(stringlist)
display_classification_report(model_summary, figure_path, figure_name)





