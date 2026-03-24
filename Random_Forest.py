
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import random

from Loader import load_one_file
from Features import feature_2, feature_3, feature_4, feature_5


# Specify the fraction of data to be used for training
fraction_train = 0.5


def random_forest(fraction_train, seed=42):
    """Creates random forest classifier, trains it and evaluates it on the test set"""
    # Generate numbers used for training and testing
    upper = 500       # Number of point cloud files
    random.seed(seed)
    train_numbers = list(random.sample(range(0,upper),int(upper*fraction_train)))
    test_numbers = [number for number in range(0,upper) if number not in train_numbers]

    # Create objects for training set
    # Initialize empty arrays
    x_train = np.zeros((len(train_numbers),4))
    y_train = np.zeros(len(train_numbers),dtype=int)
    # Iterate over the specified files to get the data
    for index, number in enumerate(train_numbers):
        data = load_one_file(number)
        # Use the selected features
        features = [feature_2(data), feature_3(data), feature_4(data), feature_5(data)]
        x_train[index] = features
        # Get the ground truth label
        y_train[index] = number//100

    # Create objects for test set
    x_test = np.zeros((len(test_numbers),4))
    y_test = np.zeros(len(test_numbers),dtype=int)
    for index, number in enumerate(test_numbers):
        data = load_one_file(number)
        # Use the selected features
        features = [feature_2(data), feature_3(data), feature_4(data), feature_5(data)]
        x_test[index] = features
        y_test[index] = number//100

    # Feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Initialize and fit the classifier with the hyperparameters
    classifier = RandomForestClassifier(n_estimators=200,           # Number of trees in the forest                                     (start with 100, 200 best)
                                        criterion='entropy',        # Function used to measure split quality ('gini' or 'entropy')      (start with gini, entropy best)
                                        max_depth= 25,              # Maximum depth of each tree                                        (start with None, tied with 25)
                                        min_samples_split=2,        # Minimum samples required to split a node.                         (start with 2)
                                        min_samples_leaf=1,         # Minimum samples required to be at a leaf node.                    (start with 1 then 2 and 5)
                                        max_features='sqrt',        # Number of features considered for splitting at each node ('sqrt', log2 or None) (start with sqrt)
                                        bootstrap = True,           # Bootstraps enabled
                                        n_jobs=-1)                  # Number of jobs to run in parallel

    classifier.fit(x_train, y_train)

    # Predicting the test set results
    predictions = classifier.predict(x_test)
    # Simple evaluation
    accuracy = classifier.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    return accuracy, predictions, y_test


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

"""# Run model
accuracy,predictions,y_test = random_forest(0.5)

# Make confusion matrix and make heatmap
labels = ["building", "car", "fence", "pole", "tree"]
cm = confusion_matrix(y_test,predictions,normalize='true')
sns.heatmap(cm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()"""

# Make learning curve
learning_curve(random_forest)