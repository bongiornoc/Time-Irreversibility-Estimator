# Time Irreversibility Estimator

The `TimeIrreversibilityEstimator` is a Python package designed to estimate time irreversibility in time series using gradient boosting classification. This package leverages the power of `xgboost` to classify forward and backward trajectories, providing a measure of time irreversibility.

## Key Features

- **Quantification of Irreversibility:** Measures the time irreversibility in high-dimensional time series using a model-free, non-linear approach.
- **Interaction Constraints:** Allows the specification of interaction constraints to explore the contribution of different feature interactions to irreversibility.
- **Cross-Validation:** Implements k-fold and group k-fold cross-validation for a robust and reliable estimation.
- **Trajectory Encoding:** Supports any encoding strategy for time series data allowing for ad-hoc hypothesis testing.


## Installation

You can install the package via pip:

```bash
pip install time_irreversibility_estimator
```

## Concept

The method introduced in the accompanying paper, "Unveiling the Drivers of Irreversibility in Time Series via Machine Learning," leverages gradient boosting to quantify the time irreversibility in high-dimensional time series. The approach rephrases the problem as a binary classification task where the direction of time (forward or backward) is to be determined. The time irreversibility measure is derived from the classifier's performance, specifically from the average log differences of predicted probabilities for forward and backward trajectories.

## Usage

Below is an example of how to use the `TimeIrreversibilityEstimator`:

```python
import time_irreversibility_estimator as ie
import numpy as np

# Example forward data (encodings of forward trajectories from a drifted 5-dimensional random-walk)
x_forward = np.random.normal(0.6, 1, size=(10000, 5))

# Example backward data (encodings of backward trajectories), optional
x_backward = -x_forward[:, ::-1]

# Example interaction constraints: '[[0, 1], [2, 3, 4]]'
# This means that features 0 and 1 can interact with each other, and features 2, 3, and 4 can interact with each other.
interaction_constraints = '[[0, 1], [2, 3, 4]]'

estimator = ie.TimeIrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
irreversibility_value = estimator.fit_predict(x_forward, x_backward)

print(f"Estimated time irreversibility: {irreversibility_value}")

# Example with GroupKFold
groups = np.random.randint(0, 5, size=x_forward.shape[0])  # Example group indices (use a meaningful group assignment here)
estimator = ie.TimeIrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
irreversibility_value = estimator.fit_predict(x_forward, x_backward, n_splits=5, groups=groups)

print(f"Estimated time irreversibility with GroupKFold: {irreversibility_value}")
```

## Class Details

### `TimeIrreversibilityEstimator`

A class to estimate time irreversibility in time series using gradient boosting classification.

#### Attributes:
- `max_depth` (int): Maximum depth of the trees in the gradient boosting model.
- `n_estimators` (int): Number of trees in the gradient boosting model.
- `learning_rate` (float): Step size shrinkage used in update of the gradient boosting model.
- `early_stopping_rounds` (int): Number of rounds for early stopping.
- `verbose` (bool): If True, print progress messages.
- `interaction_constraints` (str): Constraints on interactions between features.
- `random_state` (int or None): Seed for random number generator.

#### Methods:
- `train(self, x_forward_train, x_backward_train, x_forward_test=None, x_backward_test=None)`: Trains the model on the training set with optional test set early stopping and returns the trained model.
- `evaluate(self, model, x_forward, x_backward, return_log_diffs=False)`: Evaluates the model on some data and returns the irreversibility.
- `fit_predict(self, x_forward, x_backward=None, n_splits=5, groups=None, return_log_diffs=False)`: Performs k-fold or group k-fold cross-validation to estimate time irreversibility.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! If you would like to contribute to the code or to request a new feature, please contact christian.bongiorno@centralesupelec.fr.

## Contact

If you have any questions or feedback, please contact Christian Bongiorno at christian.bongiorno@centralesupelec.fr or Michele Vodret at mvodret@gmail.com.

## Acknowledgements

This package uses the following libraries:
- `numpy`
- `scikit-learn`
- `xgboost`

## Citation

If you use this package in your research, please cite our paper:

```
@article{Vodret2024Irreversibility,
  title={Unveiling the Drivers of Irreversibility in Time Series via Machine Learning},
  author={Michele Vodret, Cristiano Pacini, Christian Bongiorno},
  journal={In Preparation},
  year={2024},
  volume={},
  pages={},
}
```

## Authors and Contributors

- **Cristiano Pacini** 
- **Michele Vodret** - Email: [mvodret@gmail.com](mailto:mvodret@gmail.com)
- **Christian Bongiorno** (code author) - Email: [christian.bongiorno@centralesupelec.fr](mailto:christian.bongiorno@centralesupelec.fr)
