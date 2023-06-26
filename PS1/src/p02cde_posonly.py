import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # Part (c): Train and test on true labels
    model = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    # Make sure to save outputs to pred_path_c
    np.savetxt(pred_path_c, model.predict(x_test) > 0.5, delimiter='\n')
    util.plot(x_test,y_test,model.theta_0, 'incomplete1.png')
    
    # Part (d): Train on y-labels and test on true labels
    model = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    # Make sure to save outputs to pred_path_d
    np.savetxt(pred_path_d, model.predict(x_test) > .5, delimiter='\n')
    util.plot(x_test,y_test,model.theta_0, 'incomplete2.png')


    # Part (e): Apply correction factor using validation set and test on true labels
    model = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    alpha = np.mean(model.predict(x_valid))
    np.savetxt(pred_path_e, model.predict(x_test)/alpha > .5 ,delimiter='\n')
    correction = -np.log(2/alpha - 1)
    util.plot(x_test,y_test,model.theta_0, 'incomplete3.png', correction=correction)

    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
