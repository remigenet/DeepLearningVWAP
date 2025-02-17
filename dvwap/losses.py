from keras import ops

# Double casting is done here in order to work in any setup with any backend as some are
# more strict than others - you may want to remove it if you are sure of your setup

def quadratic_vwap_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    vwap_achieved = ops.sum(y_pred[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_pred[..., 0], axis = 1)
    vwap_mkt = ops.sum(y_true[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_true[..., 0], axis = 1)
    vwap_diff = vwap_achieved / vwap_mkt - 1.
    loss = ops.mean(ops.square(vwap_diff))
    return loss

def absolute_vwap_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    vwap_achieved = ops.sum(y_pred[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_pred[..., 0], axis = 1)
    vwap_mkt = ops.sum(y_true[..., 0] * y_true[..., 1], axis = 1) / ops.sum(y_true[..., 0], axis = 1)
    vwap_diff = vwap_achieved / vwap_mkt - 1.
    loss = ops.mean(ops.abs(vwap_diff))
    return loss

def volume_curve_loss(y_true, y_pred):
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.cast(y_pred, y_true.dtype)
    volume_curve_achieved = y_pred[..., 0] / ops.sum(y_pred[..., 0], axis = 1, keepdims=True)
    volume_curve_mkt = y_true[..., 0] / ops.sum(y_true[..., 0], axis = 1, keepdims=True)
    volume_curve_diff = volume_curve_achieved - volume_curve_mkt
    loss = ops.mean(ops.square(volume_curve_diff))
    return loss