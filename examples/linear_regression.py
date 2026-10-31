import numpy as np
from DLFS.algorithms.linear_regression import train


def main():

    # Generate syntehtic data
    X = None
    y = None

    # Train the model
    model, trainer = train(X=X, y=y, epochs=100, batch_size=16, learning_rate=0.01)

    # Check results
    print("TODO...Print training results")

    # Do some prediction examples...


if __name__ == "__main__":
    main()
