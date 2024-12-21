import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# Load the dataset
file_path = '/content/Room_Occupancy_Data.csv'  # Google Colab file path

data = pd.read_csv(file_path)

# Dropping Date and Time columns (not directly useful for SVM classification)
data_cleaned = data.drop(columns=['Date', 'Time'])

# Convert Room_Occupancy_Count to binary classification (0 = unoccupied, 1 = occupied)
data_cleaned['Room_Occupancy_Count'] = (data_cleaned['Room_Occupancy_Count'] > 0).astype(int)

# Separate features and target
X = data_cleaned.drop(columns=['Room_Occupancy_Count'])
y = data_cleaned['Room_Occupancy_Count']

# Normalizing features for better SVM performance
X_normalized = (X - X.min()) / (X.max() - X.min())

def fitnessFunction(k, n, trainDataset, testDataset):
    model = SVC(kernel=k, C=1/n, gamma='scale')
    model.fit(trainDataset.iloc[:, :-1], trainDataset.iloc[:, -1])
    predictions = model.predict(testDataset.iloc[:, :-1])
    accuracy = accuracy_score(testDataset.iloc[:, -1], predictions)
    return accuracy

def optimize_svm(X, y, iterations=100):
    bestAccuracy = 0
    bestKernel = ""
    bestNu = 0
    kernelList = ['rbf', 'poly', 'linear', 'sigmoid']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    trainDataset = pd.DataFrame(X_train)
    trainDataset[y.name] = y_train

    testDataset = pd.DataFrame(X_test)
    testDataset[y.name] = y_test

    convergence_data = []

    for i in range(iterations):
        k = random.choice(kernelList)
        n = random.uniform(0.01, 1)

        accuracy = fitnessFunction(k, n, trainDataset, testDataset)

        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestKernel = k
            bestNu = n

        convergence_data.append((i, bestAccuracy, bestKernel, bestNu))

    output_directory = '.'  # Google Colab working directory
    print(f'Current working directory: {output_directory}')

    iterations_list, accuracy_list = zip(*[(x[0], x[1]) for x in convergence_data])
    plt.plot(iterations_list, accuracy_list)
    plt.xlabel('Iteration')
    plt.ylabel('Best Accuracy')
    plt.title('Convergence Graph')
    plt.grid(True)

    convergence_graph_path = os.path.join(output_directory, 'convergence_graph.png')
    plt.savefig(convergence_graph_path)
    plt.close()
    print(f'Convergence graph saved at: {convergence_graph_path}')

    results_df = pd.DataFrame(convergence_data, columns=['Iteration', 'Best Accuracy', 'Best Kernel', 'Best Nu'])
    csv_file_path = os.path.join(output_directory, 'SVM_Convergence_Data.csv')
    results_df.to_csv(csv_file_path, index=False)
    print(f'Results saved at: {csv_file_path}')

    print(f'Best Accuracy: {bestAccuracy:.4f}')
    print(f'Best SVM Parameters: Kernel={bestKernel}, Nu={bestNu:.4f}')

# Run optimization on the Room Occupancy dataset
optimize_svm(X_normalized, y)
