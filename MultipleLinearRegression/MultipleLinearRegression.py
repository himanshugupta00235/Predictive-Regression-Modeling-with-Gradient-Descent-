#Importing necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

plt.style.use('dark_background')

def feature_scale(X):
    """Input : X - a numpy matrix
       Output: Scaled version of X(numpy matrix)"""
    
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    X = (X-means)/stds
    return X, means, stds

def split_data(X, Y, fraction):
    """Input : X - Features, a numpy matrix
               Y - Target, a numpy array
               fraction - a floating point value in (0, 1), for splitting data into training and testing parts
       Output: X_train - subset of X, a numpy matrix
               Y_train - subset of Y, a numpy array
               X_test - subset of X, a numpy matrix
               Y_test - subset of Y, a numpy array"""
    
    rng = np.arange(X.shape[0])
    np.random.shuffle(rng)

    numberOfTestingSamples = int(X.shape[0]*fraction)
    testingSampleIndices = rng[ : numberOfTestingSamples]
    trainingSampleIndices = rng[numberOfTestingSamples : ]

    X_train = X[trainingSampleIndices]
    Y_train = Y[trainingSampleIndices]
    X_test = X[testingSampleIndices]
    Y_test = Y[testingSampleIndices]

    return X_train, Y_train, X_test, Y_test

def compute_cost(X, Y, b, b0):
    """Input : X - Input Features, a numpy matrix
               Y - Target Variable, a numpy array
               b - a numpy array, coefficients of feature
               b0 - a floating point value, intercept of the Line
       Output: Cost of the model"""
    
    m = X.shape[0]

    predictions = X @ b + b0       # Vector of predictions
    errors = predictions - Y       # Vector of errors
    cost = (errors @ errors) / (2 * m)

    return cost

def compute_gradient(X, Y, b, b0):
    """Input : X - Input Features, a numpy matrix
               Y - Target Variable, a numpy array
               b - a numpy array, coefficients of feature
               b0 - a floating point value, intercept of the Line
       Output: dJ_db - a numpy array- differential coefficients in gradient descent
               dJ_db0 - a floating point number - differential coefficient in gradient descent"""
    
    m = X.shape[0]

    predictions = X @ b + b0        # Vector of predictions
    errors = predictions - Y        # Vector of errors
    dJ_db = (X.T @ errors) / m      # Gradient w.r.t coefficients
    dJ_db0 = np.sum(errors) / m     # Gradient w.r.t intercept

    return dJ_db, dJ_db0
    

def gradient_descent(X, Y, initial_b, initial_b0, alpha, iterations):
    """Input : X - Input Features, a numpy matrix
               Y - Target Variable, a numpy array
               initial_b - a numpy array, coefficients of feature
               initial_b0 - a floating point value, intercept of the Line
               alpha - Learning Rate
               iterations - Number of iterations to be performed
       Output: b - Ideal value of b (the coefficients of features)
               b0 - Ideal value of b0 (the intercept)
               cost_history - History of cost values throughout the iterations
               b_history - History of b values throughout the iterations
               b0_history - History of b0 values throughout the iterations"""
    
    m = X.shape[0]

    cost_history = []
    b_history = []
    b0_history = []
    b = initial_b
    b0 = initial_b0
    for i in range(iterations):
        dJ_db, dJ_db0 = compute_gradient(X, Y, b, b0)

        b = b - alpha*dJ_db
        b0 = b0 - alpha*dJ_db0

        if((i+1)%10 == 0):
            b_history.append(b)
            b0_history.append(b0)
            cost_history.append(compute_cost(X, Y, b, b0))
        
    return b, b0, cost_history, b_history, b0_history


def MultipleLinearRegression(X, Y):
    """Input : X - Input Feature, a numpy matrix
               Y - Target Variable, a numpy array
       Output: b - Ideal value of b (the coefficients of features)
               b0 - Ideal value of b0 (the intercept)
               cost_history - History of cost values throughout the iterations
               b_history - History of b values throughout the iterations
               b0_history - History of b0 values throughout the iterations"""
    
    initial_b = np.zeros(X.shape[1])
    initial_b0 = 0

    #Total Sum of Squares (TSS) or Initial cost (J):
    print(f"Initial cost (J), with b = {initial_b} and b0 = {initial_b0} : {compute_cost(X, Y, initial_b, initial_b0)}")

    #Run Gradient descent
    iterations = 10000
    alpha = 0.01

    b, b0, cost_history, b_history, b0_history = gradient_descent(X, Y, initial_b, initial_b0, alpha, iterations)

    return b, b0, cost_history, b_history, b0_history

def Plot_Results(X_train, Y_train, X_test, Y_test, b, b0, cost_history, b_history, b0_history):
    print(f"\nFinal value of b (the coefficients) = {b}")
    print(f"Final value of b0 (the intercept) = {b0:.3f} \n\n")
    training_samples = X_train.shape[0]
    testing_samples = X_test.shape[0]
    Y_train_predicted = np.zeros(training_samples)
    Y_test_predicted = np.zeros(testing_samples)

    #calculating Predicted Values
    for i in range(training_samples):
        Y_train_predicted[i] = np.dot(b, X_train[i]) + b0

    for i in range(testing_samples):
        Y_test_predicted[i] = np.dot(b, X_test[i]) + b0

    #Mean Absolute Error for training and testing data
    mae_training = np.mean(np.abs(Y_train_predicted-Y_train))
    mae_testing = np.mean(np.abs(Y_test_predicted-Y_test))
    
    #R^2
    tss_train = np.sum((Y_train - np.mean(Y_train))**2)
    rss_train = np.sum((Y_train - Y_train_predicted)**2)
    r2_train = 1 - rss_train/tss_train
    tss_test = np.sum((Y_test - np.mean(Y_test))**2)
    rss_test = np.sum((Y_test - Y_test_predicted)**2)
    r2_test = 1 - rss_test/tss_test

    print("Training Data Results : ")
    print(f"Mean Absolute Error = {mae_training : .3f}")
    print(f"R^2 = {r2_train}\n\n")
    print("Testing Data Results : ")
    print(f"Mean Absolute Error = {mae_testing : .3f}")
    print(f"R^2 = {r2_test}\n\n")

    it = np.arange(len(cost_history))
    it = (it+1)*10
    plt.plot(it, cost_history, color='pink')
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

def main():
    #Load dataset into a dataframe
    df = pd.read_csv('dataset.csv')

    #Separate input and output variables
    X = df.drop('Performance Index', axis=1).values
    Y = df['Performance Index'].values

    #Feature scale the input variable(feature)
    X, means, stds = feature_scale(X)

    #Split the data into trainig and testing parts
    np.random.seed(37)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.2)

    #Run Multiple Linear Regression
    b, b0, cost_history, b_history, b0_history = MultipleLinearRegression(X_train, Y_train)

    #Print and Plot Results
    Plot_Results(X_train, Y_train, X_test, Y_test, b, b0, cost_history, b_history, b0_history)

main()