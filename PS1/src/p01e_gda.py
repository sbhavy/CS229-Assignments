import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    x_eval, _ = util.load_dataset(eval_path, add_intercept=False)
    np.savetxt(pred_path, model.predict(x_eval) >= 0.5, delimiter='\n')
    theta = np.concatenate((np.array([model.theta_0]), model.theta))
    util.plot(x_train,y_train,theta, f'gda{int(train_path[-11])}.png')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.2, max_iter=100, eps=0.00001, theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, eps, theta_0, verbose)
    
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***

        indicator = np.int64(y > 0)

        phi = np.sum(y) / y.shape[0]

        mu = np.zeros((2, x.shape[1]))
        for i in range(x.shape[0]):
            mu[indicator[i]] += x[i]
        mu[0] /= y.shape[0] - np.sum(y)
        mu[1] /= np.sum(y)
        
        sigma = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            sigma += np.dot((x[i] - mu[indicator[i]]).reshape(-1,1), (x[i] - mu[indicator[i]]).reshape(-1,1).T)

        sigma /= x.shape[0]

        inv_sigma = np.linalg.inv(sigma)
        self.theta = inv_sigma.dot(mu[1] - mu[0])
        self.theta_0 = (mu[0] + mu[1]).T.dot(inv_sigma.dot(mu[0] - mu[1]))/2 - np.log((1-phi)/phi)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta) + self.theta_0))
        # *** END CODE HERE
