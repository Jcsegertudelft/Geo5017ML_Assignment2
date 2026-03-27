import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import random

from Loader import load_one_file
from Features import feature_2, feature_3, feature_4, feature_5


# Specify the fraction of data to be used for training
fraction_train = 0.5

def random_forest(fraction_train=0.7, seed=42):
    """Creates random forest classifier, trains it and evaluates it on the test set"""
    train_data = pd.read_csv('train_set.csv')
    test_data = pd.read_csv('test_set.csv')

    total_data = pd.concat([train_data, test_data], ignore_index=True)
    column_names = total_data.columns
    random.seed(seed)
    selection = random.sample(range(500),int(np.round((1-fraction_train)*500)))  # Random selection of test set
    test_set_ind = selection

    total_data = np.array(total_data)
    test_data = total_data[test_set_ind]
    train_data = np.delete(total_data, test_set_ind, axis=0)
    test_data = pd.DataFrame(test_data, columns=column_names)
    train_data = pd.DataFrame(train_data, columns=column_names)

    feature_cols = [col for col in train_data.columns if 'feature' in col]
    feature_cols.sort()
    x_train = np.array(train_data[feature_cols])
    x_test = np.array(test_data[feature_cols])

    y_train = np.array(train_data['class'])
    y_test = np.array(test_data['class'])


    # Initialize and fit the classifier with the hyperparameters
    classifier = RandomForestClassifier(n_estimators=200,           # Number of trees in the forest                                     (start with 100, 200 best)
                                        criterion='entropy',        # Function used to measure split quality ('gini' or 'entropy')      (start with gini, entropy best)
                                        max_depth= 25,              # Maximum depth of each tree                                        (start with None, tied with 25)
                                        min_samples_split=2,        # Minimum samples required to split a node.                         (start with 2)
                                        min_samples_leaf=1,         # Minimum samples required to be at a leaf node.                    (start with 1 then 2 and 5)
                                        max_features='sqrt',        # Number of features considered for splitting at each node ('sqrt', log2 or None) (start with sqrt)
                                        bootstrap = True,           # Bootstraps enabled
                                        n_jobs=-1,                  # Number of jobs in parallel
                                        random_state=seed)          # RNG Seed

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
    plt.savefig('learning_curve.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_cm(y_test, predictions):
    labels = ["building", "car", "fence", "pole", "tree"]
    cm = confusion_matrix(y_test, predictions, normalize='true')
    sns.heatmap(cm,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix_rf.png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    accuracy, predictions, y_test = random_forest(0.7)
    # Confusion Matrix
    create_cm(y_test, predictions)
    # Learning Curve
    learning_curve(random_forest)