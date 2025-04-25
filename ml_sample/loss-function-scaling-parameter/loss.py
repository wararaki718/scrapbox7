import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cosine_similarity(x_q: np.ndarray, x_d: np.ndarray) -> float:
    return np.dot(x_q, x_d.T) / (np.linalg.norm(x_q) * np.linalg.norm(x_d))


def loss_engagement(x_q: np.ndarray, x_d: np.ndarray, y: np.ndarray, scale: float=20.0) -> float:
    """
    Loss function for engagement prediction.
    :param x_q: Query features
    :param x_d: Document features
    :param y: Engagement labels
    :return: Loss value
    """
    c = sigmoid(scale * cosine_similarity(x_q, x_d))
    
    # Calculate the loss using binary cross-entropy
    loss = -(y * np.log(c) + (1 - y) * np.log(1 - c))
    
    # return loss
    return np.mean(loss)


def loss_relevance(x_q: np.ndarray, x_d: np.ndarray, scale: float=20.0) -> float:
    """
    Loss function for relevance prediction.
    :param x_q: Query features
    :param x_d: Document features
    :param y: Relevance labels
    :return: Loss value
    """
    c = np.exp(scale * cosine_similarity(x_q, x_d))
    
    # Calculate the loss using binary cross-entropy
    loss = - np.log(c / np.sum(c))
    
    # return loss
    return np.mean(loss)


def softmax(x_q: np.ndarray, x_d: np.ndarray, scale: float=20.0) -> np.ndarray:
    x = sigmoid(cosine_similarity(x_q, x_d))
    e_x = np.exp(scale * x)
    return e_x / np.sum(e_x, axis=0)
