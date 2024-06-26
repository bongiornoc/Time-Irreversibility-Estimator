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
    
    def train_and_evaluate(self, x_forward, x_backward, train_index, test_index):
        """
        Trains the model and evaluates it on the test set for a single fold.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray): Encodings of the backward trajectories.
        train_index (ndarray): Indices for the training set.
        test_index (ndarray): Indices for the test set.
        
        Returns:
        float: Calculated irreversibility for the fold.
        """
        y_train = np.r_[np.ones_like(train_index), np.zeros_like(train_index)]
        y_test = np.r_[np.ones_like(test_index), np.zeros_like(test_index)]

        X_train = np.row_stack((x_forward[train_index], x_backward[train_index]))
        X_test = np.row_stack((x_forward[test_index], x_backward[test_index]))

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            interaction_constraints=self.interaction_constraints,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state
        )

        if self.verbose:
            print(f"Training model on fold with train size {len(train_index)} and test size {len(test_index)}")

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=self.verbose)

        prob = model.predict_proba(X_test)[:, 1]

        irreversibility = np.log(prob)[y_test == 1].mean() - np.log(prob)[y_test == 0].mean()
        if self.verbose:
            print(f"Irreversibility of the fold: {irreversibility}")

        return irreversibility
    
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

