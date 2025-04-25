import numpy as np

from loss import sigmoid, cosine_similarity, loss_engagement, loss_relevance, softmax


def main() -> None:
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    y = sigmoid(x)
    print(y)

    x_q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_d = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    similarity = cosine_similarity(x_q, x_d)
    print(f"Cosine similarity: {similarity}")

    y = np.array([[1.0], [0.0]])
    loss = loss_engagement(x_q, x_d, y)
    print(f"Loss: {loss}")

    loss = loss_relevance(x_q, x_d)
    print(f"Loss: {loss}")
    print()

    softmax_result = softmax(x_q, x_d, scale=1.0)
    print(f"Softmax result(scale=1)   : {softmax_result.flatten()}")

    softmax_result = softmax(x_q, x_d, scale=0.01)
    print(f"Softmax result(scale=0.01): {softmax_result.flatten()}")

    softmax_result = softmax(x_q, x_d, scale=50.0)
    print(f"Softmax result(scale=50)  : {softmax_result.flatten()}")

    print("DONE")


if __name__ == "__main__":
    main()
