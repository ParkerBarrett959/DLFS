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

    # Standardize features for stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    # Standardize outputs for stability
    y_mean = y.mean()
    y_std = y.std()
    y_standardized = (y - y_mean) / y_std

    # Train the model
    model, trainer = train(
        X=X_standardized,
        y=y_standardized,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
    )

    # Predictions
    predictions_standardized = model.forward(X_standardized)
    predictions = predictions_standardized * y_std + y_mean

    # Convert weights and biases from standardized to original units
    weights = model.weights.data * y_std / X_std
    bias = y_mean + y_std * model.bias.data[0] - np.sum(weights * X_mean)

    # Evaluation
    mse = np.mean((predictions - y) ** 2)

    # Results
    print("\nTraining results:")
    # print(f"Final loss: {trainer.history[-1]['loss']}:.4f")
    print(f"True weights: {w_true}")
    print(f"True bias: {b_true}")
    print(f"Learned weights: {weights}")
    print(f"Learned bias: {bias}")


if __name__ == "__main__":
    main()
