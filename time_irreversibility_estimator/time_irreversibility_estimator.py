import numpy as np
from sklearn.model_selection import KFold, GroupKFold
import xgboost as xgb
import warnings

class TimeIrreversibilityEstimator:
    """
    A class to estimate time irreversibility in time series using gradient boosting classification.
    
    Attributes:
    
    max_depth (int): Maximum depth of the trees in the gradient boosting model.
    n_estimators (int): Number of trees in the gradient boosting model.
    early_stopping_rounds (int): Number of rounds for early stopping.
    verbose (bool): If True, print progress messages. Default is False.
    interaction_constraints (str): Constraints on interactions between features as a string.
    random_state (int or None): Seed for random number generator. Default is None.
    
    Methods:
    - train(self, x_forward_train, x_backward_train, x_forward_test=None, x_backward_test=None): Trains the model on the training set with optional test set early stopping and returns the trained model.
    - evaluate(self, model, x_forward, x_backward, return_log_diffs=False): Evaluates the model on some data and returns the irreversibility.
    - fit_predict(self, x_forward, x_backward=None,n_splits=5, groups=None, return_log_diffs=False): Performs k-fold or group k-fold cross-validation to estimate time irreversibility.
    
    Example:
    ```python
    import time_irreversibility_estimator as ie
    import numpy as np

    # Example forward data (encodings of forward trajectories)
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
    groups = np.random.randint(0, 5, size=x_forward.shape[0])  # Example group indices
    estimator = ie.TimeIrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
    irreversibility_value = estimator.fit_predict(x_forward, x_backward, n_splits=5, groups=groups)

    print(f"Estimated time irreversibility with GroupKFold: {irreversibility_value}")
    ```
    """
    
    def __init__(self, max_depth=6, n_estimators=10000, learning_rate=0.3, early_stopping_rounds=10, verbose=False, interaction_constraints=None, random_state=None,store=False):
        """
        Initializes the TimeIrreversibilityEstimator with specified parameters.
        
        Args:
        max_depth (int): Maximum depth of the trees in the gradient boosting model. Default is 6.
        n_estimators (int): Number of trees in the gradient boosting model. Default is 10000.
        learning_rate (float): Step size shrinkage used in update of the gradient boosting model. Default is 0.3.
        early_stopping_rounds (int): Number of rounds for early stopping. Default is 10.
        verbose (bool): If True, print progress messages. Default is False.
        interaction_constraints (str, optional): Constraints on interactions between features in the form of a string. For example, '[[0, 1], [2, 3, 4]]' means that features 0 and 1 can interact with each other, and features 2, 3, and 4 can interact with each other. Default is None.
        random_state (int or None): Seed for random number generator. Default is None.
        store (bool): If True, store the models, the encodings and the indices . Default is False.
        """
    
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.interaction_constraints = interaction_constraints
        self.random_state = random_state
        self.store = store
        
    def _prepare_data(self, q_forward=None,x_forward=None, x_backward=None,encoding_fun=None):
        """
        Prepares the forward and backward datasets.

        Args:
            q_forward (ndarray): 3-dimensional numpy array where axis 0 represents different trajectories and axis 1 represents the time index of each trajectory and axis 2 represents dimension of each trajectory.
            x_backward (ndarray, optional): 2-dimensional numpy array of encodings for the backward trajectories. If None, it is computed by reversing `q_forward` along axis 1 or by appling the encoding function. Default is None.
            encoding_fun (function, optional): Function to encode the trajectories. Default is None.

        Returns:
            tuple: Prepared forward and backward datasets. Each element of the tuple is a 2-dimensional numpy array where axis 0 represents different trajectories and axis 1 represents the encoding dimension of each trajectory.
        """

        if q_forward is not None:
            if len(q_forward.shape) == 2:
                raise ValueError("q_forward is 2-dimensional, it should be 3-dimensional. Use np.expand_dims(q_forward,-1) to expand the dimension.  Also, check the correct behaviour of the encoding function after that.")

            if (x_backward is not None) or (x_forward is not None):
                warnings.warn('if q_forward is provided. Both x will be ignored.')
            if encoding_fun is None:
                encoding_fun = lambda x: x.flatten()
            x_forward = np.vectorize(encoding_fun,signature='(n,m)->(k)')(q_forward)
            q_backward = q_forward[:,::-1]
            x_backward = np.vectorize(encoding_fun,signature='(n,m)->(k)')(q_backward)

        elif (x_backward is None) or (x_forward is None):
            raise ValueError("Either q_forward or x_forward and x_backward should be provided.")
        else:
            if len(x_forward.shape) != 2 or len(x_backward.shape) != 2:
                raise ValueError("x_forward and x_backward should be 2-dimensional.")
            if encoding_fun is not None:
                warnings.warn('Encoding function is provided but it will not be used since x_forward and x_backward are provided.')

        if self.store==True:
            self.x_forward = x_forward
            self.x_backward = x_backward
            self.fold_train_indices = []
            self.fold_test_indices = []
            self.models = []
            self.estimated_probabilities = []
            self.individual_irreversibility = np.zeros(len(x_forward))

        return x_forward, x_backward
    
    def train(self, x_forward_train, x_backward_train, x_forward_test=None, x_backward_test=None):
        """
        Trains the model on the training set and returns the trained model.
        
        Args:
        x_forward_train (ndarray): Encodings of the forward trajectories in the training set.
        x_backward_train (ndarray): Encodings of the backward trajectories in the training set.
        x_forward_test (ndarray, optional): Encodings of the forward trajectories in the test set. Default is None.
        x_backward_test (ndarray, optional): Encodings of the backward trajectories in the test set. Default is None.
        
        Returns:
        XGBClassifier: Trained XGBoost model.
        """
        y_train = np.r_[np.ones_like(x_forward_train), np.zeros_like(x_backward_train)]
        X_train = np.row_stack((x_forward_train, x_backward_train))

        if x_forward_test is not None and x_backward_test is not None:
            y_test = np.r_[np.ones_like(x_forward_test), np.zeros_like(x_backward_test)]
            X_test = np.row_stack((x_forward_test, x_backward_test))


        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            interaction_constraints=self.interaction_constraints,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state
        )
        model.base_score = 0.5
        
        if self.verbose:
            print(f"Training model with train size {len(x_forward_train)}")

        if x_forward_test is not None and x_backward_test is not None:
            model.fit(X_train, y_train, verbose=self.verbose, eval_set=[(X_test, y_test)])
            # if the early stopping rounds are not reached, and the last iteration is equal the number of trees, write a warning
            if model.get_booster().num_boosted_rounds() == self.n_estimators:
                warnings.warn('Early stopping rounds not reached. Consider increasing the number of trees.')
        else:
            model.fit(X_train, y_train, verbose=self.verbose)
            # Warning that use xgboost without early stopping can lead to overfitting, say to specify the test set
            warnings.warn('Early stopping rounds not specified. Consider specifying the test set for early stopping.')
        
        if self.store==True:
            self.models.append(model)

        return model
    
    def evaluate(self, model, x_forward, x_backward):
        """
        Evaluates the model on the test set and returns the time irreversibility.
        
        Args:
        model: Any model compatible with the scikit-learn API.
        x_forward (ndarray): Encodings of the forward trajectories.
        x_backward (ndarray): Encodings of the backward trajectories.
        return_log_diffs (bool): If True, return the individual log differences of the probabilities. Default is False.
        
        Returns:
        float: Calculated time irreversibility for the test set.
        list: Individual log differences of the probabilities, if return_log_diffs is True.
        """
        y_test = np.r_[np.ones(len(x_forward)), np.zeros(len(x_backward))]
        X_test = np.vstack((x_forward, x_backward))

        prob = model.predict_proba(X_test)[:, 1]
        if self.store==True:
            self.estimated_probabilities.append(prob)

        log_diffs = np.log(prob)[y_test == 1] - np.log(prob)[y_test == 0]
        irreversibility = log_diffs.mean()
        
        if self.verbose:
            print(f"Time irreversibility of the test set: {irreversibility}")

        return irreversibility, log_diffs
        
    
    def _train_and_evaluate(self, x_forward, x_backward, train_index, test_index):
        """
        Trains the model and evaluates it on the test set for a single fold.
        
        Args:
        x_forward (ndarray): Encodings of the forward trajectories. 2-dimensional numpy array where axis 0 represents different trajectories and axis 1 represents the encoding dimension of each trajectory.
        x_backward (ndarray): Encodings of the backward trajectories. 2-dimensional numpy array where axis 0 represents different trajectories and axis 1 represents the encoding dimension of each trajectory.
        train_index (ndarray): Indices for the training set.
        test_index (ndarray): Indices for the test set.
        return_log_diffs (bool): If True, return the individual log differences of the probabilities. Default is False.
        
        Returns:
        float: Calculated time irreversibility for the fold.
        list: Individual log differences of the probabilities, if return_log_diffs is True.
        """
        model = self.train(x_forward[train_index], x_backward[train_index], x_forward[test_index], x_backward[test_index])
        return self.evaluate(model, x_forward[test_index], x_backward[test_index])
    
    def fit_predict(self, q_forward=None,x_forward=None, x_backward=None,encoding_fun=lambda x: x.flatten(),  n_splits=5, groups=None):
        """
        Performs k-fold or group k-fold cross-validation to estimate time irreversibility.
        
        Args:
        forward_data (ndarray): 3-dimensional forward trajectories or 2-dimensional encoded trajectories. Axis 0 represents different trajectories, if the position are provided axis 1 represents the time and axis 2 represents the series dimension. If the encoding is provided then axis 1 is the econding dimension.
        x_backward (ndarray, optional): Encodings of the backward trajectories. If None, it is computed by reversing x_forward along axis 1. Default is None.
        encoding_fun (function, optional): Function to encode the trajectories. Default is None.
        n_splits (int): Number of folds for cross-validation. Default is 5.
        groups (array-like, optional): Group labels for the samples used while splitting the dataset into train/test set (see sklearn.model_selection.GroupFold). Default is None.
        return_log_diffs (bool): If True, return the individual log differences of the probabilities. Default is False.
        
        Returns:
        float: Mean time irreversibility over all folds.
        array: Individual log differences of the probabilities, if return_log_diffs is True.
        """
        x_forward, x_backward = self._prepare_data(q_forward,x_forward, x_backward,encoding_fun)



        if groups is not None:
            kf = GroupKFold(n_splits).split(x_forward, groups=groups)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(x_forward)
        
        D = np.zeros(n_splits)

        
        for fold_idx, (train_index, test_index) in enumerate(kf):
            if self.store==True:
                self.fold_train_indices.append(train_index)
                self.fold_test_indices.append(test_index)
            if self.verbose:
                print(f"Processing fold {fold_idx + 1}/{n_splits}")
            if self.store==True:
                D[fold_idx], self.individual_irreversibility[test_index] = self._train_and_evaluate(x_forward, x_backward, train_index, test_index)
            else:
                D[fold_idx],_ = self._train_and_evaluate(x_forward, x_backward, train_index, test_index)
        
        if self.verbose:
            print(f"Completed cross-validation with mean time irreversibility: {D.mean()}")

        return D.mean()