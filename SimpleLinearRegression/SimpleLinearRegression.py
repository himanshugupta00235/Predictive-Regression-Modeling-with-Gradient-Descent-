#Importing necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

plt.style.use('dark_background')

def scale_feature(X):
    """Input : X - a numpy array
       Output: Scaled version of X(numpy array)"""
    
    mean = np.mean(X)
    std = np.std(X)

    X = (X-mean)/std
    return X

def split_data(X, Y, fraction):
    """Input : X - Feature, a numpy array
               Y - Target, a numpy array
               fraction - a floating point value in (0, 1), for splitting data into training and testing parts
       Output: X_train - subset of X, a numpy array
               Y_train - subset of Y, a numpy array
               X_test - subset of X, a numpy array
               Y_test - subset of Y, a numpy array"""
    
    rng = np.arange(len(X))
    np.random.shuffle(rng)

    numberOfTestingSamples = int(len(X)*fraction)
    testingSampleIndices = rng[ : numberOfTestingSamples]
    trainingSampleIndices = rng[numberOfTestingSamples : ]

    X_train = X[trainingSampleIndices]
    Y_train = Y[trainingSampleIndices]
    X_test = X[testingSampleIndices]
    Y_test = Y[testingSampleIndices]

    return X_train, Y_train, X_test, Y_test

def compute_cost(X, Y, b1, b0):
    """Input : X - Input Feature, a numpy array
               Y - Target Variable, a numpy array
               b1 - a floating point value, coefficient of feature
               b0 - a floating point value, intercept of the Line
       Output: Cost of the model"""
    
    m = len(X)
    cost = (np.sum((b1*X + b0 - Y)**2)) / (2*m)

    return cost

def compute_gradient(X, Y, b1, b0):
    """Input : X - Input Feature, a numpy array
               Y - Target Variable, a numpy array
               b1 - a floating point value, coefficient of feature
               b0 - a floating point value, intercept of the Line
       Output: dJ_db1 - a floating point number - differential coefficient in gradient descent
               dJ_db0 - a floating point number - differential coefficient in gradient descent"""
    
    m = len(X)
    dJ_db1 = 0
    dJ_db0 = 0

    dJ_db1 = np.sum((b1*X + b0 - Y)*X)
    dJ_db0 = np.sum(b1*X + b0 - Y)
    
    dJ_db1 /= m
    dJ_db0 /= m

    return dJ_db1, dJ_db0
    

def gradient_descent(X, Y, initial_b1, initial_b0, alpha, iterations):
    """Input : X - Input Feature, a numpy array
               Y - Target Variable, a numpy array
               initial_b1 - a floating point value, coefficient of feature
               initial_b0 - a floating point value, intercept of the Line
               alpha - Learning Rate
               iterations - Number of iterations to be performed
       Output: b1 - Ideal value of b1
               b0 - Ideal value of b0
               cost_history - History of cost values throughout the iterations
               b1_history - History of b1 values throughout the iterations
               b0_history - History of b0 values throughout the iterations"""
    
    m = len(X)

    cost_history = []
    b1_history = []
    b0_history = []
    b1 = initial_b1 
    b0 = initial_b0
    for i in range(iterations):
        dJ_db1, dJ_db0 = compute_gradient(X, Y, b1, b0)

        b1 = b1 - alpha*dJ_db1
        b0 = b0 - alpha*dJ_db0

        if((i+1)%10 == 0):
            b1_history.append(b1)
            b0_history.append(b0)
            cost_history.append(compute_cost(X, Y, b1, b0))
        
    return b1, b0, cost_history, b1_history, b0_history


def SimpleLinearRegression(X, Y):
    """Input : X - Input Feature, a numpy array
               Y - Target Variable, a numpy array
       Output: b1 - Ideal value of b1
       b0 - Ideal value of b0
       cost_history - History of cost values throughout the iterations
       b1_history - History of b1 values throughout the iterations
       b0_history - History of b0 values throughout the iterations"""
    
    initial_b1 = 0
    initial_b0 = 0

    #Initial cost (J):
    print(f"Initial cost (J), with b1 = {initial_b1} and b0 = {initial_b0} : {compute_cost(X, Y, initial_b1, initial_b0)}")

    #Run Gradient descent
    iterations = 10000
    alpha = 1e-2

    b1, b0, cost_history, b1_history, b0_history = gradient_descent(X, Y, initial_b1, initial_b0, alpha, iterations)

    return b1, b0, cost_history, b1_history, b0_history

def Plot_Results(X_train, Y_train, X_test, Y_test, b1, b0, cost_history, b1_history, b0_history):
    print(f"\nFinal value of b1 (the coefficient) = {b1:.3f}")
    print(f"Final value of b0 (the intercept) = {b0:.3f} \n\n")
    training_samples = len(X_train)
    testing_samples = len(X_test)
    Y_train_predicted = np.zeros(training_samples)
    Y_test_predicted = np.zeros(testing_samples)

    #calculating Predicted Values
    for i in range(training_samples):
        Y_train_predicted[i] = b1*X_train[i] + b0

    for i in range(testing_samples):
        Y_test_predicted[i] = b1*X_test[i] + b0

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

    plt.scatter(X_train, Y_train, label='Actual Values')
    plt.plot(X_train, Y_train_predicted, color='#FF0000', label='Predicted Values')
    plt.title("Training Data")
    plt.xlabel("Previous Scores (Normalised)")
    plt.ylabel("Performance Index")
    plt.legend()
    plt.show()

    plt.scatter(X_test, Y_test, label='Actual Values')
    plt.plot(X_test, Y_test_predicted, color='#FF0000', label='Predicted Values')
    plt.title("Testing Data")
    plt.xlabel("Previous Scores (Normalised)")
    plt.ylabel("Performance Index")
    plt.legend()
    plt.show()

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
    X = df['Previous Scores'].values
    Y = df['Performance Index'].values

    #Feature scale the input variable(feature)
    X = scale_feature(X)

    #Split the data into trainig and testing parts
    np.random.seed(37)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, 0.2)

    #Run Simple Linear Regression
    b1, b0, cost_history, b1_history, b0_history = SimpleLinearRegression(X_train, Y_train)

    #Print and Plot Results
    Plot_Results(X_train, Y_train, X_test, Y_test, b1, b0, cost_history, b1_history, b0_history)

main()