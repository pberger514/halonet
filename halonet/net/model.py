#
# A keras implementation of a 3D CNN
#
# Authors: P. Berger, G. Stein
#
# References: 
# [1] V-Net https://arxiv.org/abs/1606.04797
# [2] V-Net in pytorch https://github.com/mattmacy/vnet.pytorch
# [3] Cosmology https://arxiv.org/pdf/1711.02033.pdf
# [4] YOLO (You Only Look Once)?
# 
#

#
# Things to check / add:
# - Dropout at fully connected layers
# - Average pooling vs. Max pooling
# - Batch normalization
# - Non-sequential (split and then collect)
# - Leaky ReLU
# - Strides in the convolution layers
#

import keras.models as km
import keras.layers as kl
from keras.layers.merge import add


#######################################################################


def get_model(nlevels, nfmi = 16, input_shape = (64, 64, 64, 1)):

    """
    Returns a CNN V-Net Keras model.

    Parameters
    -----------
        nlevels: list
            Number of levels in the V-Net, which is the number of times the resolution of
            of the filter is halved.

        nfmi: int 
            The number of feature maps in the first convolutional layer (default = 16).

        intput_shape: tuple, 
            the shape of the input data, 
            default = (64, 64, 64, 1)

    Output
    -------
        model: 
            the constructed Keras model.

    """

    # Initialize the model.
    #model = km.Sequential()
    #This is not a Sequential model so create input state
    input_state = kl.Input(shape = input_shape)

    #These are hard coded following [1]
    conv_kernel_size = (5, 5, 5)
    updown_kernel_size = (2, 2, 2)
    updown_stride = updown_kernel_size

    level_list = []

    for fi in range(nlevels):
        # Add the initial convolution layer, with nfm feature maps.
        # For simplicity we are doing only one convolution per level (see [1]).
        if fi == 0:
            residual = kl.Conv3D(nfm, kernel_size = conv_kernel_size, padding='same')(input_state)
            shortcut = kl.Conv3D(nfm, kernel_size = (1, 1, 1), padding='same')(input_state)

        else:
            residual = kl.Conv3D(nfm*2**fi, kernel_size = conv_kernel_size, padding='same')(x)
            shortcut = kl.Conv3D(nfm*2**fi, kernel_size = (1, 1, 1), padding='same')(x)

        # Perform elementwise sum with input to train on residuals.
        x = add([residual, shortcut])

        #Save a copy for fine-grained features forwarding
        level_list.append(add([residual, shortcut]))

        # Peform a down convolution with PReLU activation
        x = kl.Conv3D(nfm, kernel_size = updown_kernel_size, strides=updown_stride,
                      activation = kl.PReLU, padding='same')(x) #padding?

        # Check average pooling vs. residual network?

    # Add a fully-connected layer?
    #model.add(kl.Dense(numnodes, activation = 'tanh'))

    # Step back up to achieve initial resolution
    for fi in range(nlevels)[::-1]:
        
        if fi != nlevels:
            #Concatenate with fine grained forwarded features along filters axis
            x = kl.Concatenate(axis=-1)([level_list[fi], x])

        #Perform a couple convolutions
        residual = kl.Conv3D(nfm*2**(fi-1), kernel_size = conv_kernel_size, padding='same')(x)
        residual = kl.Conv3D(nfm*2**(fi-1), kernel_size = conv_kernel_size, padding='same')(residual)
        shortcut = kl.Conv3D(nfm*2**(fi-1), kernel_size = (1, 1, 1), padding='same')(x)
        x = add([residual, shortcut])
            
        # Peform an up-convolution with PReLU activation
        x = kl.Conv3DTranspose(nfm, kernel_size = updown_kernel_size, strides=updown_stride,
                               activation = kl.PReLU)(x)

    
    model = km.Model(inputs = input_state, outputs = x)

    return model
