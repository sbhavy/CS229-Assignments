import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    min_tau_pos = -1
    min_MSE = 100000000
    # Search tau_values for the best tau (lowest MSE on the validation set)
    for i, tau in enumerate(tau_values):
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)

        x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
        y_pred = model.predict(x_valid)
        MSE = np.mean((y_valid - y_pred)**2)
        print(MSE)

        plt.figure()
        plt.plot(x_train, y_train, 'bx', linewidth=2)
        plt.plot(x_valid, y_pred, 'rx', linewidth=2)
        plt.savefig(f'p05_tau{i+1}.png')

        if MSE < min_MSE:
            min_MSE = MSE
            min_tau_pos = i

    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(tau=tau_values[min_tau_pos])
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    MSE = np.mean((y_test - y_pred)**2)
    print(tau_values[min_tau_pos], MSE)
    # Save predictions to pred_path
    np.savetxt(pred_path, y_pred, delimiter='\n')
    # Plot data
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_pred, 'rx', linewidth=2)
    plt.savefig(f'p05_tautest.png')    
    # *** END CODE HERE ***
