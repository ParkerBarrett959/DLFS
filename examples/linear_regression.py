import numpy as np
from DLFS.algorithms.linear_regression import train


def main():

    # Generate synthetic feature data - this model takes house square footage and bedroom size and predicts price.
    sample_size = 1000
    square_footage = np.random.normal(loc=2000.0, scale=500.0, size=sample_size)
    bedrooms = np.random.normal(loc=3.0, scale=1.0, size=sample_size)
    X = np.column_stack((square_footage, bedrooms))

    # Gemerate true weights and bias
    w_true = np.array([150.0, 20000.0])
    b_true = 30000.0

    # Generate training data noise
    noise = np.random.normal(loc=0, scale=15000.0, size=sample_size)

    # Generate synthetic output data (home price)
    y = X @ w_true + b_true + noise

    # Train the model
    model, trainer = train(X=X, y=y, epochs=100, batch_size=16, learning_rate=0.01)

    # Check results
    print("TODO...Print training results")

    # Do some prediction examples...


if __name__ == "__main__":
    main()
