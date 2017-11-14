#
# A keras implementation of a 3D CNN
#
# Authors: P. Berger, G. Stein
#
# References: 
# - V-Net https://arxiv.org/abs/1606.04797
# - V-Net in pytorch https://github.com/mattmacy/vnet.pytorch
# - Cosmology https://arxiv.org/pdf/1711.02033.pdf
# - YOLO (You Only Look Once)?
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


#######################################################################


def get_model(filters=[2, 10], numfm, numnodes, input_shape = (64, 64, 64, 1),
              output_size = (64, 64, 64)):

    """
    Returns a CNN V-Net Keras model.

    Parameters
    -----------
        filters: list
            Number of layers/filter banks to use.

        numfm: int 
            the number of feature maps in the convolution layer

        numnodes: int
            the number of nodes in the fully-connected layer

        intput_shape: tuple, 
            the shape of the input data, 
            default = (64, 64, 64, 1)

        output_size: int, 
            the number of nodes in the output layer,
            default = 10.

    Output
    -------
        model: 
            the constructed Keras model.

    """

    # Initialize the model.
    model = km.Sequential()

    for fi, filt in enumerate(filters):
        # Add a 2D convolution layer, with numfm feature maps.
        if fi == 0:
            model.add(kl.Conv3D(numfm, kernel_size = (filt, filt, filt),
                                input_shape = input_shape,
                                activation = 'relu'))
        else:
            model.add(kl.Conv3D(numfm, kernel_size = (filt, filt, filt),
                                activation = 'relu'))

        # Add an average pooling layer.
        #model.add(kl.AvergePooling3D(pool_size = (2, 2, 2),
        #                             strides = (2, 2, 2)))


    # Convert the network from 3D to 1D.
    #model.add(kl.Flatten())

    # Add a fully-connected layer.
    #model.add(kl.Dense(numnodes, activation = 'tanh'))

    for fi, filt in enumerate(filters[::-1]):
        # Add a 2D convolution layer, with numfm feature maps.
        model.add(kl.Conv3DTranspose(numfm, kernel_size = (filt, filt, filt),
                                     activation = 'relu'))


    # Return the model.
    return model

