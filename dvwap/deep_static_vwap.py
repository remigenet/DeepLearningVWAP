import keras
from keras import ops
from keras.models import Model
from tln import TLN
BACKEND = keras.backend.backend()
if BACKEND == 'tensorflow':
    import tensorflow as tf

@keras.utils.register_keras_serializable(package='dvwap', name='StaticVWAP')
class StaticVWAP(Model):
    def __init__(self, lookback, n_ahead, input_include_aheads = False, internal_model = None,  *args, **kwargs):
        super(StaticVWAP, self).__init__(*args, **kwargs)
        self.lookback = lookback
        self.n_ahead = n_ahead
        self.input_include_aheads = input_include_aheads 
        self.internal_model = internal_model if internal_model is not None else TLN(output_len = n_ahead,
                            output_features = 1,
                            flatten_output = False,
                            hidden_layers = 2,
                            use_convolution = True,
                            name = 'TLN')
        self.add_dims_to_internal_model = None
        self.dictionnary_input = None

    def build(self, input_shape):
        if isinstance(input_shape, dict):
            feature_shape = input_shape['features']
            self.dictionnary_input = True
        else:
            feature_shape = input_shape
            self.dictionnary_input = False
        internal_model_input_shape = (feature_shape[0], self.lookback, feature_shape[2])
        internal_model_target_output_shape = (feature_shape[0], self.n_ahead, 1)
        self.internal_model.build(internal_model_input_shape)
        internal_model_output_shape = self.internal_model.compute_output_shape(internal_model_input_shape)
        if internal_model_output_shape != internal_model_target_output_shape:
            if (*internal_model_output_shape, 1) ==  internal_model_target_output_shape:
                self.add_dims_to_internal_model = True
            else:
                raise ValueError(f"""internal model do not respect required output shape:
                Received inputs shape {internal_model_input_shape} 
                Return outputs shape {internal_model_output_shape}
                Required outputs shape {internal_model_target_output_shape}
                """)
        else:
            self.add_dims_to_internal_model = False

    def call(self, inputs):
        if self.dictionnary_input:
            features = inputs['features']
        else:
            features = inputs
        if self.input_include_aheads:
            if BACKEND == 'tensorflow':
                features = tf.slice(features, [0, 0, 0], [tf.shape(features)[0], self.lookback, features.shape[2]])
            else:
                features = ops.slice(features, (0,0,0), (features.shape[0], self.lookback, features.shape[2]))
        preds = self.internal_model(features)
        if self.add_dims_to_internal_model:
            preds = ops.expand_dims(preds, axis=-1)
        volume_curve = ops.softmax(preds, axis=1)
        results = ops.concatenate([volume_curve, ops.zeros_like(preds)], axis=-1)
        return results

    def get_config(self):
        config = super(StaticVWAP, self).get_config()
        config.update({
            'lookback': self.lookback,
            'n_ahead': self.n_ahead,
            'input_include_aheads': self.input_include_aheads,
            'internal_model': keras.saving.serialize_keras_object(self.internal_model),
        })
        return config

    @classmethod
    def from_config(cls, config):
        internal_model_config = config.pop('internal_model')
        internal_model = keras.saving.deserialize_keras_object(internal_model_config)
        return cls(internal_model=internal_model, **config)

    def get_build_config(self):
        return {
            'add_dims_to_internal_model': self.add_dims_to_internal_model,
            'dictionnary_input': self.dictionnary_input
        }

    def build_from_config(self, config):
        self.add_dims_to_internal_model = config.get('add_dims_to_internal_model')
        self.dictionnary_input = config.get('dictionnary_input')
