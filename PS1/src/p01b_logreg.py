import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    np.savetxt(pred_path, model.predict(x_eval) >= 0.5, delimiter='\n')
    util.plot(x_train,y_train,model.theta_0, f'logistic{int(train_path[-11])}.png')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.2, max_iter=100, eps=0.00001, theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, eps, theta_0, verbose)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.theta_0 = np.zeros(np.shape(x)[1])
        theta_old = None

        while True:

            h = 1. / (1. + np.exp(-x.dot(self.theta_0)))

            grad_J = np.dot(x.T, (h - y)) / np.shape(x)[0]
            hessian_J =np.dot(x.T * h * (1 - h), x) / np.shape(x)[0]

            delta = np.dot(np.linalg.inv(hessian_J), grad_J)
            theta_old = self.theta_0.copy()
            self.theta_0 -= delta

            if np.linalg.norm(self.theta_0 - theta_old) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta_0)))
        # *** END CODE HERE ***
