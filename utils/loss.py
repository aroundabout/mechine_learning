import keras.backend as K


def symmetrical_cross_entropy(y_true, y_pred):
    loss = -0.5 * K.sum(y_true * K.log(y_pred) + y_pred * K.log(y_true), axis=-1)
    return loss


def cross_entropy(y_true, y_pred):
    loss = -K.sum(y_true * K.log(y_pred), axis=-1)
    return loss


def fusion_cross_entropy(y_true, y_pred):
    return (symmetrical_cross_entropy(y_true, y_pred)+cross_entropy(y_true, y_pred))/2
