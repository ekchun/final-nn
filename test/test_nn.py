# TODO: import dependencies and write unit tests below
from nn.nn import NeuralNetwork as nn
from nn.io import read_text_file, read_fasta_file
from nn.preprocess import one_hot_encode_seqs,  sample_seqs   

import numpy as np
import numpy.testing as npt
import pytest

test_arch = [{"input_dim": 4, "output_dim": 2, "activation": "relu"},
             {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}]


def test_single_forward():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")

    A_prev = np.random.randn(1, 2)       # shape (1, 2)
    W = np.random.randn(3, 2)            # (D_out=3, D_prev=2)
    b = np.random.randn(3, 1)            # (3,1)
    activation = "relu"

    A_curr, Z_curr = model._single_forward(W, b, A_prev, activation)

    assert A_curr.shape == (1, 3)
    assert Z_curr.shape == (1, 3)

def test_forward():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mean_squared_error")

    # single example
    X = np.random.randn(1, 4)
    output, cache = model.forward(X)

    # final output should be (1,1)
    assert output.shape == (1, 1)
    assert isinstance(cache, dict)

    # cache should have A0
    assert "A0" in cache
    assert np.all(cache['A0'] == X)

def test_single_backprop():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mean_squared_error")

    W = np.random.randn(3, 2)         # (D_out, D_prev)
    b = np.random.randn(3, 1)
    Z = np.random.randn(1, 3)         # (N, D_out)
    A_prev = np.random.randn(1, 2)    # (N, D_prev)
    dA_curr = np.random.randn(1, 3)   # (N, D_out)
    activation = "relu"

    dA_prev, dW, db = model._single_backprop(W, b, Z, A_prev, dA_curr, activation)

    assert dA_prev.shape == (1, 2)   # (N, D_prev)
    assert dW.shape == (3, 2)        # (D_out, D_prev)
    assert db.shape == (3, 1)        # (D_out, 1)

def test_predict():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mean_squared_error")
    X = np.random.randn(1, 4)   # 1 example
    y_hat = model.predict(X)

    # predict returns (N, D_out)
    assert y_hat.shape == (1, 1)

def test_binary_cross_entropy():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")

    # labels and preds must be same shape (N, 1)
    y = np.array([[1.0]])
    y_hat = np.array([[0.9]])
    loss = model._binary_cross_entropy(y, y_hat)

    assert isinstance(loss, float)
    assert loss >= 0.0

def test_binary_cross_entropy_backprop():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="binary_cross_entropy")
    y = np.array([[1.0]])
    y_hat = np.array([[0.9]])
    dA = model._binary_cross_entropy_backprop(y, y_hat)

    # same shape as y
    assert dA.shape == y.shape
    # numeric sanity: clipped preds shouldn't produce NaN/inf
    assert np.all(np.isfinite(dA))

def test_mean_squared_error():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mean_squared_error")
    y = np.array([[1.0]])
    y_hat = np.array([[0.9]])
    loss = model._mean_squared_error(y, y_hat)

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_mean_squared_error_backprop():
    model = nn(test_arch, lr=0.01, seed=42, batch_size=1, epochs=10, loss_function="mean_squared_error")
    y = np.array([[1.0]])
    y_hat = np.array([[0.9]])
    dA = model._mean_squared_error_backprop(y, y_hat)

    assert dA.shape == y.shape
    assert np.all(np.isfinite(dA))

def test_sample_seqs():
    seqs = ["AAAA", "TTTT", "CCCC", "GGGG", "AAAA"]
    labels = [1, 0, 0, 0, 1]
    s_seqs, s_labels = sample_seqs(seqs, labels)
    counts = np.bincount(np.array(s_labels, dtype=int))

    assert counts[0] == counts[1]

def test_one_hot_encode_seqs():
    seqs = ["A", "T", "C", "G"]
    X = one_hot_encode_seqs(seqs)
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],], dtype=float)

    npt.assert_allclose(X, expected, atol=1e-9)
    assert X.shape == (4, 4)