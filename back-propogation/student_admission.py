import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

epochs = 1000
learnrate = 0.5


def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')


if __name__ == '__main__':
    data = pd.read_csv('student_data.csv')
    one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
    print(one_hot_data)
    one_hot_data = one_hot_data.drop('rank', axis=1)
    processed_data = one_hot_data[:]
    sample = np.random.choice(processed_data.index, size=int(len(processed_data) * 0.9), replace=False)
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
    features = train_data.drop('admit', axis=1)
    targets = train_data['admit']
    features_test = test_data.drop('admit', axis=1)
    targets_test = test_data['admit']