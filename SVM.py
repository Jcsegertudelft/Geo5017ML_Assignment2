import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from Loader import load_one_file
from Features import feature_2, feature_3, feature_4, feature_5

def svm_classifier(fraction_train, seed=42, kernel='rbf', C=1.0, gamma='scale', degree=3):
    """Creates random forest classifier, trains it and evaluates it on the test set"""
    train_data = pd.read_csv('train_set.csv')
    test_data = pd.read_csv('test_set.csv')

    total_data = pd.concat([train_data, test_data], ignore_index=True)
    column_names = total_data.columns
    random.seed(seed)
    selection = random.sample(range(500), int(np.round((1 - fraction_train) * 500)))  # Random selection of test set
    test_set_ind = selection

    total_data = np.array(total_data)
    test_data = total_data[test_set_ind]
    train_data = np.delete(total_data, test_set_ind, axis=0)
    test_data = pd.DataFrame(test_data, columns=column_names)
    train_data = pd.DataFrame(train_data, columns=column_names)

    x_train = np.array(train_data[['feature_2', 'feature_3', 'feature_4', 'feature_5']])
    x_test = np.array(test_data[['feature_2', 'feature_3', 'feature_4', 'feature_5']])

    y_train = np.array(train_data['class'])
    y_test = np.array(test_data['class'])

    # Initializing SVM here
    classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree
    )

    classifier.fit(x_train, y_train)

    # Predict and evaluate
    predictions = classifier.predict(x_test)
    accuracy = classifier.score(x_test, y_test)

    print(
        f"kernel={kernel}, C={C}, gamma={gamma}, degree={degree}, "
        f"accuracy={accuracy:.4f}"
    )

    return accuracy, predictions, y_test

def hyperparameter():
    """ Using different kernels and hyperparameters"""
    results=[]
    # Linear
    for C in [0.1,1,10]:
        accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='linear', C=C)
        results.append(('linear', C, None, None, accuracy))

    # RBF
    for C in [1,10]:
        for gamma in ['scale',0.1]:
            accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='rbf', C=C, gamma=gamma)
            results.append(('rbf', C, gamma, None, accuracy))

    # Polynomial
    for C in [0.1,1,10]:
        for gamma in ['scale','auto',0.01,0.1]:
            for degree in [2,3,4]:
                accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='poly', C=C, gamma=gamma,degree=degree)
                results.append(('poly', C, gamma, degree, accuracy))

    # sigmoid
    for C in [0.1, 1, 10]:
        for gamma in ['scale', 'auto', 0.01, 0.1]:
            accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='sigmoid', C=C, gamma=gamma)
            results.append(('sigmoid', C, gamma, None, accuracy))

    results.sort(key=lambda x: x[4], reverse=True)

    print("\nTop 5 SVM configurations:")
    for row in results[:5]:
        print(row)

    return results

def learning_curve(model):
    """Creates learning curve and plots it, using three runs per fraction"""
    training_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_accuracies = []
    seeds = [42, 43, 44]
    for training_fraction in training_fractions:
        print(f"Training fraction: {training_fraction}")
        accuracies = []
        for seed in seeds:
            accuracy,_,_ = model(training_fraction, seed)
            accuracies.append(accuracy)
        avg_accuracy = np.mean(accuracies)
        avg_accuracies.append(avg_accuracy)
        print(f"Average accuracy: {avg_accuracy}")

    # Make plot
    plt.figure(figsize=(10, 6))
    plt.plot(training_fractions, avg_accuracies, marker='o', label='Average Accuracy')
    plt.title('Learning Curve (using three runs per fraction)')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    results = hyperparameter()
    best = results[0]

    learning_curve(
        lambda fraction, seed: svm_classifier(
            fraction, seed,
            kernel=best[0],
            C=best[1],
            gamma='scale' if best[2] is None else best[2],
            degree=3 if best[3] is None else best[3]
        )
    )
