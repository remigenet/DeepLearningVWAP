import numpy as np
from keras.models import Model
from keras import ops
from typing import Union, Dict, Optional
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from dvwap import absolute_vwap_loss, quadratic_vwap_loss, volume_curve_loss

class DynamicVWAPStandardModel:
    """
    Implement Dyanmism following the paper 'Improving VWAP strategies: A dynamic volume approach',  Jedrzej Bialkowski (1) , Serge Darolles (2) , Gaëlle Le Fol (3) 
    Accepts any standard sklearn model and keras Model
    For sklearn model it will by default flatten the 2 last dimension together while keeping them for keras model - and will only use the first column of the target (volume)
    For custom keras loss that needs to use both columns of target, like the absolute or quadratic vwap loss, you need to specify loss_use_price = True when instanciating a model
    """
    def __init__(self, base_model: Union[Model, RegressorMixin], need_flat_input: Optional[bool] = None, loss_use_price: bool = False):
        if isinstance(base_model, Model) and need_flat_input is None:
            self.need_flat_input = False
        elif isinstance(base_model, RegressorMixin) and need_flat_input is None:
            self.need_flat_input = True
        elif need_flat_input is not None:
            self.need_flat_input = need_flat_input
        else:
            raise ValueError('If not providing a standard sklearn model or keras model you need to provide a value for need_flat_input')
        self.base_model = base_model
        self.loss_use_price = loss_use_price

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs):
        """
        Calibration methods:
        X should be a matrix (num_elements, lookback+n_ahead-1, features) shapes 
        y should be a matrix (num_elements, n_ahead, 2) with volumes in first column and price in the second column
        """
        self.n_ahead, self.n_features = y.shape[1], X.shape[2]
        if X.ndim != 3:
            raise ValueError('For Dynamic VWAP X should be shape (num_elements, lookback+n_ahead, features)')
        if X.shape[1] < self.n_ahead:
            raise ValueError('For Dynamic VWAP, X should include at least as many periods on sequence dimension as there is step ahead')
        X_trunc = X[:,:-self.n_ahead + 1] # + 1 is because we do not avec n_ahead value in  X but n_ahead - 1
        if self.need_flat_input:
            X_trunc = X_trunc.reshape(X.shape[0], -1)
        if not self.loss_use_price:
            y = y[:,:,0]
        self.base_model.fit(X_trunc, y, *args, **kwargs)

    def predict(self, X: np.ndarray, verbose=False):
        y = np.zeros((X.shape[0], self.n_ahead))
        remaining_proportion = np.ones(X.shape[0])
        for step in range(self.n_ahead - 1):
            X_trunc = X[:,step:-self.n_ahead + step + 1]
            if self.need_flat_input:
                X_trunc = X_trunc.reshape(X.shape[0], -1)
            if isinstance(self.base_model, Model):
                raw_preds = self.base_model.predict(X_trunc, verbose=verbose, batch_size=X.shape[0])
            else:
                raw_preds = self.base_model.predict(X_trunc)
            if self.loss_use_price:
                raw_preds = raw_preds[:,:,0]
            if step > 0:
                raw_preds=raw_preds[:,:-step]
            raw_preds = np.clip(raw_preds, a_min=0, a_max=None)
            volume_curve = np.nan_to_num(raw_preds / np.sum(raw_preds, axis = 1, keepdims=True), nan=1 / raw_preds.shape[1])
            proportion = volume_curve[:,0] * remaining_proportion
            remaining_proportion -= proportion
            y[:,step]=proportion
        y[:,-1] = remaining_proportion
        return y

    def predict_static(self, X: np.ndarray, verbose=False):
        """
        Function to create prediction without dynamism, using simply base model to predict future volume curve at time 0 and use it
        """
        X_trunc = X[:,:-self.n_ahead + 1]
        if self.need_flat_input:
            X_trunc = X_trunc.reshape(X.shape[0], -1)
        if isinstance(self.base_model, Model):
            raw_preds = self.base_model.predict(X_trunc, verbose=verbose, batch_size=X.shape[0])
        else:
            raw_preds = self.base_model.predict(X_trunc)
        if self.loss_use_price:
            raw_preds = raw_preds[:,:,0]  
        raw_preds = np.clip(raw_preds, a_min=0, a_max=None)
        volume_curve = np.nan_to_num(raw_preds / np.sum(raw_preds, axis = 1, keepdims=True), nan=1 / raw_preds.shape[1])
        return volume_curve

    def _score(self, y_pred: np.ndarray, y_true: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        preds = np.expand_dims(y_pred, axis=2)
        y_true[:,:,0] = np.nan_to_num(y_true[:,:,0] / np.sum(y_true[:,:,0], axis=1, keepdims=True), nan=1 / y_true.shape[1])
        absolute_model_score = absolute_vwap_loss(y_true, preds)
        quadratic_model_score = quadratic_vwap_loss(y_true, preds)
        model_volume_curve_score = volume_curve_loss(y_true, preds)
        r2_volume_curve = r2_score(y_true=y_true[:,:,0].flatten(), y_pred=preds.flatten())

        if verbose:
            print(f"""model train on VWAP loss:
            Score for Benchmark Execution:
            Absolute VWAP loss:
            model score: {absolute_model_score}
            ---
            Quadratic VWAP Loss:
            model score: {quadratic_model_score}
            ---
            Score for Volume Curve Proximity:
            model score: {model_volume_curve_score}
            model r2: {r2_volume_curve}
            """)
        return {
            'absolute_vwap_loss': absolute_model_score,
            'quadratic_vwap_loss': quadratic_model_score,
            'volume_curve_loss': model_volume_curve_score,
            'r2 score for volume curve': r2_volume_curve,
        }
    
    def score(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Generate prediction with X and y and returns a dictionnary with different metrics using the dynamic method
        """
        preds = self.predict(X, verbose = verbose)
        return self._score(y_pred = preds, y_true=y, verbose = verbose)    

    def score_static(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Generate prediction with X and y and returns a dictionnary with different metrics using the static method
        """
        preds = self.predict_static(X)
        return self._score(y_pred = preds, y_true=y, verbose = verbose)    