import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from Loader import load_one_file
from Features import feature_2, feature_3, feature_4, feature_5

def svm_classifier(fraction_train, seed=42, kernel='rbf', C=1.0, gamma='scale', degree=3):
    """Creates SVM classifier, trains it and evaluates it on the test set"""
    # Generate numbers used for training and testing
    upper = 500  # Number of point cloud files
    random.seed(seed)
    train_numbers = list(random.sample(range(0, upper), int(upper * fraction_train)))
    test_numbers = [number for number in range(0, upper) if number not in train_numbers]

    # Create objects for training set
    # Initialize empty arrays
    x_train = np.zeros((len(train_numbers), 4))
    y_train = np.zeros(len(train_numbers), dtype=int)
    # Iterate over the specified files to get the data
    for index, number in enumerate(train_numbers):
        data = load_one_file(number)
        # Use the selected features
        features = [feature_2(data), feature_3(data), feature_4(data), feature_5(data)]
        x_train[index] = features
        # Get the ground truth label
        y_train[index] = number // 100

    # Create objects for test set
    x_test = np.zeros((len(test_numbers), 4))
    y_test = np.zeros(len(test_numbers), dtype=int)
    for index, number in enumerate(test_numbers):
        data = load_one_file(number)
        # Use the selected features
        features = [feature_2(data), feature_3(data), feature_4(data), feature_5(data)]
        x_test[index] = features
        y_test[index] = number // 100

    # Feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

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

    # # Polynomial
    # for C in [0.1,1,10]:
    #     for gamma in ['scale','auto',0.01,0.1]:
    #         for degree in [2,3,4]:
    #             accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='poly', C=C, gamma=gamma,degree=degree)
    #             results.append(('poly', C, gamma, degree, accuracy))
    #
    # # sigmoid
    # for C in [0.1, 1, 10]:
    #     for gamma in ['scale', 'auto', 0.01, 0.1]:
    #         accuracy,_,_= svm_classifier(fraction_train=0.6,seed=42, kernel='sigmoid', C=C, gamma=gamma)
    #         results.append(('sigmoid', C, gamma, None, accuracy))

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
