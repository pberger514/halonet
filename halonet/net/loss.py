from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object

def dice_loss_coefficient(y_true, y_pred):

    return -2*K.sum(y_true*y_pred, axis=-1) / K.sum(K.square(y_true) + K.square(y_pred), axis=-1)
