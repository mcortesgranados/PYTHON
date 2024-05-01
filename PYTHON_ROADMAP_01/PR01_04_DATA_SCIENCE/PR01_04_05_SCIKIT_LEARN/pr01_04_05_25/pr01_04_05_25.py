"""
25. Bias-Variance Tradeoff Analysis: Balancing model complexity and generalization performance.

The bias-variance tradeoff is a fundamental concept in machine learning that aims to balance the complexity of a model 
with its ability to generalize to unseen data. A model with high bias may oversimplify the underlying relationships in the data (underfitting), 
while a model with high variance may capture noise in the training data (overfitting). Let's explore the bias-variance tradeoff using scikit-learn:

Explanation:

We import necessary libraries including NumPy, Matplotlib, learning_curve from scikit-learn, load_digits from sklearn.datasets, and SVC from sklearn.svm.

We load the digits dataset, which consists of handwritten digits.

We define a function plot_learning_curve to plot the learning curve of a model. The learning curve shows the training and cross-validation scores 
as a function of the number of training examples.

We create an instance of the Support Vector Classifier (SVC) with a linear kernel.

We call the plot_learning_curve function to plot the learning curve of the SVC model.

The learning curve shows two lines: one for the training score and one for the cross-validation score. The shaded areas around the lines 
represent the variance (spread) of the scores.

By analyzing the learning curve, we can assess the bias-variance tradeoff. If the training and cross-validation scores converge and 
plateau at a high value, the model has low variance and low bias. If there is a significant gap between the training and cross-validation scores, 
the model may have high variance and low bias, indicating overfitting. If both scores are low, the model may have high bias and low variance, 
indicating underfitting. The goal is to find a balance between bias and variance that maximizes performance on unseen data.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC

# Loading the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Defining a function to plot learning curves
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Creating a Support Vector Classifier (SVC) model
svc = SVC(kernel='linear')

# Plotting the learning curve
plot_learning_curve(svc, "Learning Curve (SVC)", X, y, ylim=(0.7, 1.01), cv=5, n_jobs=-1)

plt.show()
