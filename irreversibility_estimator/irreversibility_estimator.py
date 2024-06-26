import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb

class IrreversibilityEstimator:
    """
    A class to estimate irreversibility in time series using gradient boosting classification.
    
    Attributes:
    n_splits (int): Number of folds for cross-validation.
    max_depth (int): Maximum depth of the trees in the gradient boosting model.
    n_estimators (int): Number of trees in the gradient boosting model.
    early_stopping_rounds (int): Number of rounds for early stopping.
    verbose (bool): If True, print progress messages. Default is False.
    interaction_constraints (str): Constraints on interactions between features as a string.
    random_state (int or None): Seed for random number generator. Default is None.
    kf (KFold): KFold cross-validator.
    
    Methods:
    - prepare_data(self, x_forward, x_backward=None): Prepares the forward and backward datasets.
    - train(self, x_forward, x_backward, train_index): Trains the model on the training set and returns the trained model.
    - evaluate(self, model, x_forward, x_backward, test_index, return_log_diffs=False): Evaluates the model on the test set and returns the irreversibility.
    - train_and_evaluate(self, x_forward, x_backward, train_index, test_index, return_log_diffs=False): Trains the model and evaluates it on the test set for a single fold.
    - fit_predict(self, x_forward, x_backward=None): Performs k-fold cross-validation to estimate irreversibility.
    
    Example:
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
    """
    
    def __init__(self, n_splits=5, max_depth=6, n_estimators=10000, early_stopping_rounds=10, verbose=False, interaction_constraints=None, random_state=None):
        """
        Initializes the IrreversibilityEstimator with specified parameters.
        
        Args:
        n_splits (int): Number of folds for cross-validation. Default is 5.
        max_depth (int): Maximum depth of the trees in the gradient boosting model. Default is 6.
        n_estimators (int): Number of trees in the gradient boosting model. Default is 10000.
        early_stopping_rounds (int): Number of rounds for early stopping. Default is 10.
        verbose (bool): If True, print progress messages. Default is False.
        interaction_constraints (str, optional): Constraints on interactions between features in the form of a string. For example, '[[0, 1], [2, 3, 4]]' means that features 0 and 1 can interact with each other, and features 2, 3, and 4 can interact with each other. Default is None.
        random_state (int or None): Seed for random number generator. Default is None.
        """
        self.n_splits = n_splits
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.interaction_constraints = interaction_constraints
        self.random_state = random_state
        self.kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
    def prepare_data(self, x_forward, x_backward=None):
        """
        Prepares the forward and backward datasets.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray, optional): Encodings of the backward trajectories. If None, it is computed by reversing x_forward along axis 1. Default is None.
        
        Returns:
        tuple: Prepared forward and backward datasets.
        """
        if x_backward is None:
            x_backward = x_forward[:, ::-1]
        return x_forward, x_backward
    
    def train(self, x_forward, x_backward, train_index):
        """
        Trains the model on the training set and returns the trained model.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray): Encodings of the backward trajectories.
        train_index (ndarray): Indices for the training set.
        
        Returns:
        XGBClassifier: Trained XGBoost model.
        """
        y_train = np.r_[np.ones_like(train_index), np.zeros_like(train_index)]
        X_train = np.row_stack((x_forward[train_index], x_backward[train_index]))

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            interaction_constraints=self.interaction_constraints,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state
        )

        if self.verbose:
            print(f"Training model with train size {len(train_index)}")

        model.fit(X_train, y_train, verbose=self.verbose)
        
        return model
    
    def evaluate(self, model, x_forward, x_backward, test_index, return_log_diffs=False):
        """
        Evaluates the model on the test set and returns the irreversibility.
        
        Args:
        model (XGBClassifier): Trained XGBoost model.
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray): Encodings of the backward trajectories.
        test_index (ndarray): Indices for the test set.
        return_log_diffs (bool): If True, return the individual log differences of the probabilities. Default is False.
        
        Returns:
        float: Calculated irreversibility for the test set.
        list: Individual log differences of the probabilities, if return_log_diffs is True.
        """
        y_test = np.r_[np.ones_like(test_index), np.zeros_like(test_index)]
        X_test = np.row_stack((x_forward[test_index], x_backward[test_index]))

        prob = model.predict_proba(X_test)[:, 1]

        log_diffs = np.log(prob)[y_test == 1] - np.log(prob)[y_test == 0]
        irreversibility = log_diffs.mean()
        
        if self.verbose:
            print(f"Irreversibility of the test set: {irreversibility}")

        if return_log_diffs:
            return irreversibility, log_diffs
        else:
            return irreversibility
    
    def train_and_evaluate(self, x_forward, x_backward, train_index, test_index, return_log_diffs=False):
        """
        Trains the model and evaluates it on the test set for a single fold.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray): Encodings of the backward trajectories.
        train_index (ndarray): Indices for the training set.
        test_index (ndarray): Indices for the test set.
        return_log_diffs (bool): If True, return the individual log differences of the probabilities. Default is False.
        
        Returns:
        float: Calculated irreversibility for the fold.
        list: Individual log differences of the probabilities, if return_log_diffs is True.
        """
        model = self.train(x_forward, x_backward, train_index)
        return self.evaluate(model, x_forward, x_backward, test_index, return_log_diffs)
    
    def fit_predict(self, x_forward, x_backward=None):
        """
        Performs k-fold cross-validation to estimate irreversibility.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray, optional): Encodings of the backward trajectories. If None, it is computed by reversing x_forward along axis 1. Default is None.
        
        Returns:
        float: Mean irreversibility over all folds.
        """
        x_forward, x_backward = self.prepare_data(x_forward, x_backward)
        D = np.zeros(self.n_splits)
        
        for fold_idx, (train_index, test_index) in enumerate(self.kf.split(x_forward)):
            if self.verbose:
                print(f"Processing fold {fold_idx + 1}/{self.n_splits}")
            D[fold_idx] = self.train_and_evaluate(x_forward, x_backward, train_index, test_index)
        
        if self.verbose:
            print(f"Completed cross-validation with mean irreversibility: {D.mean()}")

        return D.mean()