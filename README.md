
# Irreversibility Estimator

The `IrreversibilityEstimator` is a Python package designed to estimate irreversibility in time series using gradient boosting classification. This package leverages the power of `xgboost` to classify forward and backward trajectories, providing a measure of irreversibility.

## Key Features

- **Cross-Validation:** Utilizes k-fold cross-validation to ensure robust estimation.
- **Customizable Model:** Allows customization of `xgboost` parameters, including maximum tree depth, number of estimators, and interaction constraints.
- **Data Preparation:** Handles preparation of forward and backward datasets, including optional automatic generation of backward trajectories.

## Installation

You can install the package via pip:

```bash
pip install irreversibility_estimator
```

## Usage

Below is an example of how to use the `IrreversibilityEstimator`:

```python
from irreversibility_estimator.irreversibility_estimator import IrreversibilityEstimator
import numpy as np

# Example forward data (encodings of forward trajectories)
x_forward = np.random.normal(0.6, 1, size=(10000, 5))

# Example backward data (encodings of backward trajectories), optional
x_backward = -x_forward[:, ::-1]

# Example interaction constraints: '[[0, 1], [2, 3, 4]]'
# This means that features 0 and 1 can interact with each other, and features 2, 3, and 4 can interact with each other.
interaction_constraints = '[[0, 1], [2, 3, 4]]'

estimator = IrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
irreversibility_value = estimator.fit_predict(x_forward, x_backward)

print(f"Estimated irreversibility: {irreversibility_value}")
```

## Class Details

### `IrreversibilityEstimator`

A class to estimate irreversibility in time series using gradient boosting classification.

#### Attributes:
- `n_splits` (int): Number of folds for cross-validation.
- `max_depth` (int): Maximum depth of the trees in the gradient boosting model.
- `n_estimators` (int): Number of trees in the gradient boosting model.
- `early_stopping_rounds` (int): Number of rounds for early stopping.
- `verbose` (bool): If True, print progress messages.
- `interaction_constraints` (str): Constraints on interactions between features.
- `random_state` (int or None): Seed for random number generator.
- `kf` (KFold): KFold cross-validator.

#### Methods:
- `__init__(self, n_splits=5, max_depth=6, n_estimators=10000, early_stopping_rounds=10, verbose=False, interaction_constraints=None, random_state=None)`: Initializes the estimator with specified parameters.
- `prepare_data(self, x_forward, x_backward=None)`: Prepares the forward and backward datasets.
- `train_and_evaluate(self, x_forward, x_backward, train_index, test_index)`: Trains the model and evaluates it on the test set for a single fold.
- `fit_predict(self, x_forward, x_backward=None)`: Performs k-fold cross-validation to estimate irreversibility.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please contact [Your Name] at [your.email@example.com].

## Acknowledgements

This package uses the following libraries:
- `numpy`
- `scikit-learn`
- `xgboost`

