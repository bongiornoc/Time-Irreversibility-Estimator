import numpy as np
from sklearn.model_selection import KFold, GroupKFold, train_test_split
import xgboost as xgb
import warnings

class TimeIrreversibilityEstimator:
    """
    A class to estimate time irreversibility in time series using gradient boosting classification.

    Attributes:
    -----------
    max_depth : int
        Maximum depth of the trees in the gradient boosting model.
    n_estimators : int
        Number of trees in the gradient boosting model.
    learning_rate : float
        Step size shrinkage used in update to prevent overfitting.
    early_stopping_rounds : int
        Number of rounds for early stopping.
    verbose : bool
        If True, print progress messages. Default is False.
    interaction_constraints : str
        Constraints on interactions between features as a string.
    random_state : int or None
        Seed for random number generator. Default is None.
    store : bool
        If True, store the models, the encodings, the indices, the probabilities, and the individual irreversibility values. Default is False.
    kwargs : dict
        Additional parameters to be passed to the XGBoost classifier

    Methods:
    --------
    
    train(self, x_forward_train, x_backward_train, x_forward_test=None, x_backward_test=None)
        Trains the model on the training set and returns the trained model.
    
    evaluate(self, model, x_forward, x_backward)
        Evaluates the model on the test set and returns the time irreversibility and individual log differences of the probabilities.
    
    fit_predict(self, q_forward=None, x_forward=None, x_backward=None, encoding_fun=lambda x: x.flatten(), n_splits=5, groups=None)
        Performs k-fold or group k-fold cross-validation to estimate time irreversibility.

    Example:
    --------
    ```python
    import numpy as np
    from time_irreversibility_estimator import TimeIrreversibilityEstimator

    # Example forward forward trajectories
    q_forward = np.random.normal(0.6, 1, size=(10000, 6,1)).cumsum(axis=1)

    # Example of encoding function
    encoding_fun = lambda x: np.diff(x,axis=0)

    # Example interaction constraints: '[[0, 1], [2, 3, 4]]'
    interaction_constraints = '[[0, 1], [2, 3, 4]]'

    estimator = TimeIrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
    irreversibility_value = estimator.fit_predict(q_forward=q_forward, encoding_fun=encoding_fun)

    print(f"Estimated time irreversibility: {irreversibility_value}")

    # Example with GroupKFold
    groups = np.random.randint(0, 5, size=q_forward.shape[0])  # Example group indices
    estimator = TimeIrreversibilityEstimator(interaction_constraints=interaction_constraints, verbose=True, random_state=0)
    irreversibility_value = estimator.fit_predict(q_forward, n_splits=5, groups=groups, encoding_fun=encoding_fun)

    print(f"Estimated time irreversibility with GroupKFold: {irreversibility_value}")
    ```

    Citation:
    ---------
    If you use this package in your research, please cite our paper:

    ```
    @article{vodret2024functional,
        title={Functional Decomposition and Estimation of Irreversibility in Time Series via Machine Learning},
        author={Vodret, Michele and Pacini, Cristiano and Bongiorno, Christian},
        journal={arXiv preprint arXiv:2407.06063},
        year={2024}
    }
    ```
    """

    def __init__(self, max_depth=6, n_estimators=10000, learning_rate=0.3, early_stopping_rounds=10, verbose=False, 
                 interaction_constraints=None, random_state=None, store=False, **kwargs):
        """
        Initializes the TimeIrreversibilityEstimator with specified parameters.
        
        Parameters:
        -----------
        max_depth : int, optional
            Maximum depth of the trees in the gradient boosting model. Default is 6.
        n_estimators : int, optional
            Number of trees in the gradient boosting model. Default is 10000.
        learning_rate : float, optional
            Step size shrinkage used in update to prevent overfitting. Default is 0.3.
        early_stopping_rounds : int, optional
            Number of rounds for early stopping. Default is 10.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        interaction_constraints : str, optional
            Constraints on interactions between features in the form of a string. Default is None.
        random_state : int or None, optional
            Seed for random number generator. Default is None.
        store : bool, optional
            If True, store the models, the encodings, the indices, the probabilities, and the individual irreversibility values. Default is False.
        kwargs : dict
            Additional parameters to be passed to the XGBoost classifier.
        """
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.interaction_constraints = interaction_constraints
        self.random_state = random_state
        self.store = store
        self.kwargs = kwargs

    def _prepare_data(self, q_forward=None, x_forward=None, x_backward=None, encoding_fun=None):
        """
        Prepares the forward and backward datasets.

        Parameters:
        -----------
        q_forward : ndarray, optional
            3-dimensional numpy array where axis 0 represents different trajectories, axis 1 represents the time index of each trajectory, and axis 2 represents the dimension of each trajectory.
        x_forward : ndarray, optional
            2-dimensional numpy array of encodings for the forward trajectories. Default is None.
        x_backward : ndarray, optional
            2-dimensional numpy array of encodings for the backward trajectories. If None, it is computed by reversing `q_forward` along axis 1 or by applying the encoding function. Default is None.
        encoding_fun : function, optional
            Function to encode the trajectories. Default is None.

        Returns:
        --------
        tuple
            Prepared forward and backward datasets. Each element of the tuple is a 2-dimensional numpy array where axis 0 represents different trajectories and axis 1 represents the encoding dimension of each trajectory.

        Raises:
        -------
        ValueError
            If neither `q_forward` nor `x_forward` and `x_backward` are provided, or if the dimensions of the inputs are incorrect.
        """
        if q_forward is not None:
            if len(q_forward.shape) == 2:
                raise ValueError("q_forward is 2-dimensional, it should be 3-dimensional. Use np.expand_dims(q_forward, -1) to expand the dimension. Also, check the correct behavior of the encoding function after that.")

            if (x_backward is not None) or (x_forward is not None):
                warnings.warn('If q_forward is provided, both x_forward and x_backward will be ignored.')
            if encoding_fun is None:
                encoding_fun = lambda x: x.flatten()
            x_forward = np.vectorize(encoding_fun, signature='(n,m)->(k,1)')(q_forward).reshape(q_forward.shape[0], -1)
            q_backward = q_forward[:, ::-1]
            x_backward = np.vectorize(encoding_fun, signature='(n,m)->(k,1)')(q_backward).reshape(q_backward.shape[0], -1)
        elif (x_backward is None) or (x_forward is None):
            raise ValueError("Either q_forward or x_forward and x_backward should be provided.")
        else:
            if len(x_forward.shape) != 2 or len(x_backward.shape) != 2:
                raise ValueError("x_forward and x_backward should be 2-dimensional.")
            if encoding_fun is not None:
                warnings.warn('Encoding function is provided but it will not be used since x_forward and x_backward are provided.')

        if self.store:
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

        Parameters:
        -----------
        x_forward_train : ndarray
            2-dimensional encodings of the forward trajectories in the training set.
        x_backward_train : ndarray
            2-dimensional encodings of the backward trajectories in the training set.
        x_forward_test : ndarray, optional
            2-dimensional encodings of the forward trajectories in the test set. Default is None.
        x_backward_test : ndarray, optional
            2-dimensional encodings of the backward trajectories in the test set. Default is None.

        Returns:
        --------
        XGBClassifier
            Trained XGBoost model.

        Raises:
        -------
        ValueError
            If the input dimensions are incorrect or the number of trajectories does not match.
        """
        if len(x_forward_train.shape) != 2 or len(x_backward_train.shape) != 2 or (x_forward_test is not None and x_backward_test is not None and (len(x_forward_test.shape) != 2 or len(x_backward_test.shape) != 2)):
            raise ValueError("x_forward_train, x_backward_train, x_forward_test, and x_backward_test should be 2-dimensional.")
        if len(x_backward_train) != len(x_backward_train):
            raise ValueError("Number of forward and backward trajectories should be equal.")
        if x_forward_test is not None and x_backward_test is not None:
            if len(x_forward_test) != len(x_backward_test):
                raise ValueError("Number of forward and backward trajectories in the test set should be equal.")
            
        y_train = np.r_[np.ones(len(x_forward_train)), np.zeros(len(x_backward_train))]
        X_train = np.row_stack((x_forward_train, x_backward_train))

        if x_forward_test is not None and x_backward_test is not None:
            y_test = np.r_[np.ones(len(x_forward_test)), np.zeros(len(x_backward_test))]
            X_test = np.row_stack((x_forward_test, x_backward_test))

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            interaction_constraints=self.interaction_constraints,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            **self.kwargs
        )
        model.base_score = 0.5

        if self.verbose:
            print(f"Training model with train size {len(x_forward_train)}")

        if x_forward_test is not None and x_backward_test is not None:
            model.fit(X_train, y_train, verbose=self.verbose, eval_set=[(X_test, y_test)])
            if model.get_booster().num_boosted_rounds() == self.n_estimators:
                warnings.warn('Early stopping rounds not reached. Consider increasing the number of trees.')
        else:
            model.fit(X_train, y_train, verbose=self.verbose)
            warnings.warn('Early stopping rounds not specified. Consider specifying the test set for early stopping.')

        if self.store:
            self.models.append(model)

        return model

    def evaluate(self, model, x_forward, x_backward):
        """
        Evaluates the model on the test set and returns the time irreversibility and individual log differences of the probabilities.

        Parameters:
        -----------
        model : XGBClassifier
            Trained XGBoost model.
        x_forward : ndarray
            2-dimensional encodings of the forward trajectories. Axis 0 represents different trajectories and axis 1 represents the encoding dimension.
        x_backward : ndarray
            2-dimensional encodings of the backward trajectories. Axis 0 represents different trajectories and axis 1 represents the encoding dimension.

        Returns:
        --------
        float
            Calculated time irreversibility for the test set.
        list
            Individual log differences of the probabilities.

        Raises:
        -------
        ValueError
            If the input dimensions are incorrect.
        """
        if len(x_forward.shape) != 2 or len(x_backward.shape) != 2:
            raise ValueError("x_forward and x_backward should be 2-dimensional.")
        
        y_test = np.r_[np.ones(len(x_forward)), np.zeros(len(x_backward))]
        X_test = np.vstack((x_forward, x_backward))

        prob = model.predict_proba(X_test)[:, 1]
        if self.store:
            self.estimated_probabilities.append(prob)

        log_diffs = np.log(prob[y_test == 1]) - np.log(prob[y_test == 0])
        irreversibility = log_diffs.mean()

        if self.verbose:
            print(f"Time irreversibility of the test set: {irreversibility}")

        return irreversibility, log_diffs

    def _train_and_evaluate(self, x_forward, x_backward, train_index, test_index):
        """
        Trains the model and evaluates it on the test set for a single fold.

        Parameters:
        -----------
        x_forward : ndarray
            Encodings of the forward trajectories.
        x_backward : ndarray
            Encodings of the backward trajectories.
        train_index : ndarray
            Indices for the training set.
        test_index : ndarray
            Indices for the test set.

        Returns:
        --------
        float
            Calculated time irreversibility for the fold.
        list
            Individual log differences of the probabilities.
        """
        model = self.train(x_forward[train_index], x_backward[train_index], x_forward[test_index], x_backward[test_index])
        return self.evaluate(model, x_forward[test_index], x_backward[test_index])
    
    def _train_evaluate_insample(self, x_forward, x_backward, test_size=0.2):
        """
        Trains the model on the training set and evaluates it on the test set.

        Parameters:
        -----------
        x_forward : ndarray
            Encodings of the forward trajectories.
        x_backward : ndarray
            Encodings of the backward trajectories.
        test_size : float, optional
            Fraction of the dataset to include in the test split. Default is 0.2.

        Returns:
        --------
        float
            Calculated time irreversibility for the test set.
        list
            Individual log differences of the probabilities.
        """
        
        indices = np.arange(len(x_forward))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=self.random_state)
        x_forward_train, x_forward_test = x_forward[train_indices], x_forward[test_indices]
        x_backward_train, x_backward_test = x_backward[train_indices], x_backward[test_indices]

        model = self.train(x_forward_train, x_backward_train, x_forward_test, x_backward_test)
        return self.evaluate(model, x_forward_train, x_backward_train)

    def fit_predict(self, q_forward=None, x_forward=None, x_backward=None, encoding_fun=lambda x: x.flatten(), n_splits=5, groups=None):
        """
        Performs k-fold or group k-fold cross-validation to estimate time irreversibility.

        Parameters:
        -----------
        q_forward : ndarray, optional
            3-dimensional forward trajectories. Axis 0 represents different trajectories, axis 1 represents the time index of each trajectory, and axis 2 represents the dimension of each trajectory. Default is None.
        x_forward : ndarray, optional
            2-dimensional encodings of the forward trajectories. Axis 0 represents different trajectories and axis 1 represents the encoding dimension. Default is None.
        x_backward : ndarray, optional
            2-dimensional encodings of the backward trajectories. Axis 0 represents different trajectories and axis 1 represents the encoding dimension. Default is None.
        encoding_fun : function, optional
            Function to encode the trajectories. Apply the function to every trajectory q, i.e., x = [f(q[0]),f(q[1]),...,f(q[n])]. Default is lambda x: x.flatten().
        n_splits : int, optional
            Number of folds for cross-validation. Default is 5.
        groups : array-like, optional
            Group labels for the samples used while splitting the dataset into train/test sets. Default is None.

        **Note:** You cannot set all of `q_forward`, `x_forward`, and `x_backward` to None. You must provide either the 3-dimensional trajectories (`q_forward`) or both 2-dimensional encodings (`x_forward` and `x_backward`).

        Returns:
        --------
        float
            Mean time irreversibility over all folds.

        Raises:
        -------
        ValueError
            If neither `q_forward` nor `x_forward` and `x_backward` are provided, or if the dimensions of the inputs are incorrect.
        """
        x_forward, x_backward = self._prepare_data(q_forward, x_forward, x_backward, encoding_fun)

        if groups is not None:
            kf = GroupKFold(n_splits).split(x_forward, groups=groups)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(x_forward)

        D = np.zeros(n_splits)

        for fold_idx, (train_index, test_index) in enumerate(kf):
            if self.store:
                self.fold_train_indices.append(train_index)
                self.fold_test_indices.append(test_index)
            if self.verbose:
                print(f"Processing fold {fold_idx + 1}/{n_splits}")
            if self.store:
                D[fold_idx], self.individual_irreversibility[test_index] = self._train_and_evaluate(x_forward, x_backward, train_index, test_index)
            else:
                D[fold_idx], _ = self._train_and_evaluate(x_forward, x_backward, train_index, test_index)

        if self.verbose:
            print(f"Completed cross-validation with mean time irreversibility: {D.mean()}")

        return D.mean()