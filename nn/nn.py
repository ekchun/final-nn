# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union(int, str)]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        #check inputs
        if activation not in ['relu', 'sigmoid']:
            raise ValueError("activation must be either 'relu' or 'sigmoid'")
        if W_curr.shape[1] != A_prev.shape[1]:
            raise ValueError("W_curr and A_prev have incompatible shapes")
        if W_curr.shape[0] != b_curr.shape[0]:
            raise ValueError("W_curr and b_curr have incompatible shapes")
        
        #lin transform
        Z_curr = W_curr.T @ A_prev + b_curr.T

        #activation
        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            A_curr = self._sigmoid(Z_curr)
        
        return A_curr, Z_curr


    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        TODO:This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        #check inputs
        if X.shape[1] != self.arch[0]['input_dim']:
            raise ValueError("Input data has incorrect shape")
        
        cache = {}
        cache['A0'] = X
        A_prev = X

        for idx, layer in enumerate(self.arch): # loop through layers
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation_curr = layer['activation']

            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation_curr)

            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

            A_prev = A_curr # update for next layer

        output = A_curr
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        #check inputs
        if activation_curr not in ['relu', 'sigmoid']:
            raise ValueError("activation_curr must be either 'relu' or 'sigmoid'")
        if W_curr.shape[0] != dA_curr.shape[0]:
            raise ValueError("W_curr and dA_curr have incompatible shapes")
        if Z_curr.shape != dA_curr.shape:
            raise ValueError("Z_curr and dA_curr must have the same shape")
        if A_prev.shape[1] != dA_curr.shape[0]:
            raise ValueError("A_prev and dA_curr have incompatible shapes")
        
        #activation backprop
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'linear':
            dZ_curr = dA_curr
        else:
            raise ValueError("Unsupported activation type")
        
        N = dA_curr.shape[0]
        D_curr = Z_curr.shape[1]
        #gradients
        dA_prev = dZ_curr @ W_curr
        dW_curr = (1/N) * (dZ_curr.T @ A_prev)
        db_curr = (1/N) * np.sum(dZ_curr, axis = 0).reshape(D_curr, 1) # sum over all items/examples

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        TODO: This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        #check inputs
        if y.shape != y_hat.shape:
            raise ValueError("y and y_hat must have the same shape")

        N = y.shape[0]
        L = len(self.arch)
        grad_dict = {}

        #intial gradient
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        
        # loop thru layers... in reverse
        for idx in range(L-1, -1, -1):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(idx)] #A0

            activation = self.arch[idx]['activation']

            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)
            grad_dict['dW' + str(layer_idx)]
            grad_dict['db' + str(layer_idx)]

            dA_curr = dA_prev #propagate
        
        return grad_dict


    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        TODO: This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        TODO: This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        for epoch in range(self._epochs):
            for i in range(0, X_train.shape[0], self._batch_size):
                X_batch = X_train[i:i+self._batch_size]
                y_batch = y_train[i:i+self._batch_size]

                #forward pass
                y_hat, cache = self.forward(X_batch)

                #compute loss
                if self._loss_func == 'binary_cross_entropy':
                    loss_train = self._binary_cross_entropy(y_batch, y_hat)
                else:
                    loss_train = self._mean_squared_error(y_batch, y_hat)
                
                epoch_loss_sum += loss_train + X_batch.shape[0]
                epoch_seen += X_batch.shape[0]

                #backprop
                grad_dict = self.backprop(y_batch, y_hat, cache)

                #update params
                self._update_params(grad_dict)
            
            # avg loss
            train_epoch_loss = epoch_loss_sum / float(epoch_seen)
            per_epoch_loss_train.append(train_epoch_loss)

            # validation loss
            y_val_hat, _ = self.forward(X_val)
            if self._loss_func == 'binary_cross_entropy':
                loss_val = self._binary_cross_entropy(y_val, y_val_hat)
            else:
                loss_val = self._mean_squared_error(y_val, y_val_hat)
            per_epoch_loss_val.append(loss_val)

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        TODO: This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, _ = self.forward(X)

        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))

        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #check inputs
        if dA.shape != Z.shape:
            raise ValueError("dA and Z must have the same shape")
        
        sig = self._sigmoid(Z)
        dZ = dA * sig * (1 - sig)

        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.maximum(0, Z)
        
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #check inputs
        if dA.shape != Z.shape:
            raise ValueError("dA and Z must have the same shape")
        
        dZ = dA * (Z > 0)

        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #check inputs
        if y.shape != y_hat.shape:
            raise ValueError("y and y_hat must have the same shape")
        
        # #convert to float
        # y_hat = y_hat.astype(float)
        # y = y.astype(float)

        #clip y_hat for stability
        eps = 1e-9
        y_hat_clip = np.clip(y_hat, eps, 1 - eps)

        #binary cross entropy loss
        bce_loss = - (y * np.log(y_hat_clip) + (1 - y) * np.log(1 - y_hat_clip))
        # if multi-D, sum bce loss across output dimension then average
        if bce_loss.ndim > 1:
            per_ex_loss = np.sum(bce_loss, axis=1)
        else:
            per_ex_loss = bce_loss
        
        return float(np.mean(per_ex_loss))

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        #check inputs
        if y.shape != y_hat.shape:
            raise ValueError("y and y_hat must have the same shape")
        
        #bce derivative
        eps = 1e-9
        y_hat_clip = np.clip(y_hat, eps, 1 - eps)

        dA = (1 / y.shape[0]) * ((y_hat_clip - y) / (y_hat_clip * (1 - y_hat_clip))) # average over batch size

        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        #check inputs
        if y.shape != y_hat.shape:
            raise ValueError("y and y_hat must have the same shape")

        #MSE
        sq_error = (y_hat - y) ** 2
        # if multi-D, sum squared error across output dimension then average
        if sq_error.ndim > 1:
            per_ex_error = np.sum(sq_error, axis=1)
        else:
            per_ex_error = sq_error
        
        return float(np.mean(per_ex_error))

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        #check inputs
        if y.shape != y_hat.shape:
            raise ValueError("y and y_hat must have the same shape")
        
        #MSE derivative
        dA = (2 / y.shape[0]) * (y_hat - y) # average over batch size

        return dA