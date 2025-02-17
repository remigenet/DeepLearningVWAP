import os
import tempfile
BACKEND = 'jax'
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import numpy as np
import pandas as pd
import keras
from sklearn.linear_model import LinearRegression

from dvwap import StaticVWAP, absolute_vwap_loss, quadratic_vwap_loss, volume_curve_loss, DynamicVWAPStandardModel
from dvwap.data_formater import full_generate


@pytest.fixture
def model_parameters():
    return {
        'batch_size': 64,
        'epochs': 3,
        'lookback': 15,
        'n_ahead': 6,
        'target_asset': 'AAPL'
    }

@pytest.fixture
def generated_data(model_parameters):
    def generate_random_data(asset_list, num_periods):
        end_date = pd.Timestamp.now().floor('h')
        start_date = end_date - pd.Timedelta(hours=num_periods-1)
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = np.exp(np.random.rand(num_periods, len(asset_list)))
        df = pd.DataFrame(data, columns=asset_list, index=date_range)
        return df

    notionals = generate_random_data([model_parameters['target_asset']], 10000)
    volumes = generate_random_data([model_parameters['target_asset']], 10000)
    
    X_train, X_test, y_train, y_test = full_generate(
        volumes, 
        notionals,
        model_parameters['target_asset'],
        lookback=model_parameters['lookback'],
        n_ahead=model_parameters['n_ahead'],
        include_ahead_inputs=True
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture(params=[absolute_vwap_loss, quadratic_vwap_loss, volume_curve_loss])
def loss_function(request):
    return request.param


@pytest.fixture(params=['adam', 'sgd'])
def optimizer(request):
    return request.param

def test_static_vwap_fit_and_save(model_parameters, generated_data, optimizer, loss_function):
    assert keras.backend.backend() == BACKEND
    
    X_train, X_test, y_train, y_test = generated_data
    
    model = StaticVWAP(
        lookback=model_parameters['lookback'],
        n_ahead=model_parameters['n_ahead'],
        input_include_aheads=True
    )
    model.compile(optimizer=optimizer, loss=loss_function)
    
    history = model.fit(
        X_train, y_train,
        batch_size=model_parameters['batch_size'],
        epochs=model_parameters['epochs'],
        validation_split=0.2,
        shuffle=True,
        verbose=False
    )
    
    # Get predictions before saving
    predictions_before = model.predict(X_test, verbose=False)
    
    # Save and load the model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'static_model.keras')
        model.save(model_path)
        loaded_model = keras.models.load_model(model_path, custom_objects={loss_function.__name__: loss_function})
    
    # Get predictions after loading
    predictions_after = loaded_model.predict(X_test, verbose=False)
    
    # Compare predictions
    np.testing.assert_allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-8)
    
    # Test that the loaded model can be used for further training
    loaded_model.fit(X_train, y_train, epochs=1, batch_size=16, verbose=False)

    print(f"Static VWAP model with {optimizer} optimizer and {loss_function.__name__} successfully saved, loaded, and reused.")

def test_dynamic_vwap_framework_fit_with_linear_regression(model_parameters, generated_data):
    assert keras.backend.backend() == BACKEND
    
    X_train, X_test, y_train, y_test = generated_data
    
    model = DynamicVWAPStandardModel(base_model = LinearRegression())
    
    model.fit(
        X_train['features'], y_train,
    )
    
    score =  model.score(X_test['features'], y_test, verbose = False)
    score_static = model.score_static(X_test['features'], y_test, verbose = False)

def test_dynamic_vwap_framework_fit_with_linear_regression(model_parameters, generated_data, optimizer, loss_function):
    assert keras.backend.backend() == BACKEND
    
    X_train, X_test, y_train, y_test = generated_data
    
    base_model = StaticVWAP(
        lookback=model_parameters['lookback'],
        n_ahead=model_parameters['n_ahead'],
        input_include_aheads=False
    ) 
    base_model.compile(optimizer=optimizer, loss=loss_function)
    model = DynamicVWAPStandardModel(base_model = base_model, loss_use_price=True)
    
    model.fit(
        X_train['features'], y_train,
    )
    
    score =  model.score(X_test['features'], y_test, verbose = False)
    score_static = model.score_static(X_test['features'], y_test, verbose = False)