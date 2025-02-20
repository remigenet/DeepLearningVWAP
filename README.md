# Deep Learning for VWAP Execution

This repository presents Aplo's latest research on VWAP execution and contains the code discussed in the paper [Deep Learning for VWAP Execution in Crypto Markets: Beyond the Volume Curve](https://arxiv.org/abs/2502.13722).

## Model

The model is proposed as a keras3 package that works with any backend (tensorflow, jax, or torch), although we recommend using jax for optimal performance.

## Installation

1. Download the repository
2. Run `pip install .` or `poetry install` (recommended) in the root folder of the repository

## Usage

```python
from dvwap import StaticVWAP, quadratic_vwap_loss, absolute_vwap_loss, volume_curve_loss

BATCH_SIZE = 128
N_MAX_EPOCHS = 1000

# Specify your model parameters
# lookback and n_ahead are integer values representing the window sizes
model = StaticVWAP(lookback=lookback, n_ahead=n_ahead, input_include_aheads=False)

# The StaticVWAP is a keras model, so it works similarly
# To minimize the VWAP loss effectively, use a loss function that accounts for it
# These losses are available in the dvwap package
model.compile(optimizer='adam', loss=quadratic_vwap_loss)

# Training
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, 
                    validation_split=0.2, callbacks=callbacks(), shuffle=True, verbose=False)

# Prediction
preds = model.predict(X_test, verbose=False)
```

### Model Parameters

- `lookback` and `n_ahead` are integer values representing the window sizes.
- `input_include_aheads`: You can have two formats of inputs for your model:
  1. If `True`, the feature inputs should be formatted to have a size of `lookback + n_ahead` on the sequence length.
  2. If `False`, pass inputs with only `lookback` on sequence length.

Note: Future values are not used by the static model, as it gives its prediction at t0 for the full period. The `input_include_aheads` option is provided for flexibility when comparing with dynamic models that require future inputs to progressively update their predictions.

### Loss Functions

The `StaticVWAP` is a Keras model. To minimize the VWAP loss effectively, you should use a loss function that accounts for it. These loss functions are available in the `dvwap` package:

- `quadratic_vwap_loss`
- `absolute_vwap_loss`
- `volume_curve_loss`

## Data Formatting

The model expects inputs as a dictionary:
```python
{
    "prices": array shape (num_elem, seq_len, 1),
    "volumes": array shape (num_elem, seq_len, 1),
    "features": array shape (num_elem, seq_len, num_features)
}
```

Training targets should be an array of shape `(num_elem, n_ahead, 2)` with volumes in the first element of the last dimension, and price in the second.

To use the default data formatter from the paper:

```python
import pandas as pd
from dvwap.data_formater import full_generate

volumes = pd.read_parquet('path_to_your_volume_data.parquet')
notionals = pd.read_parquet('path_to_your_notionals_data.parquet')

X_train, X_test, y_train, y_test = full_generate(volumes, notionals, target_asset, 
                                                 lookback=120, n_ahead=12, test_split=0.2, 
                                                 include_ahead_inputs=False, autoscale_target=True)
```

## Other models

The package also contains the dynamic framework following the paper 'Improving VWAP strategies: A dynamic volume approach',  Jedrzej Bialkowski (1) , Serge Darolles (2) , Gaëlle Le Fol (3). It is available importing DynamicVWAPStandardModel. For more details look in the example and results section, or in the code. 

## Results

Tables and graphs of results from the paper can be found in the `example_and_results` folder.

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg