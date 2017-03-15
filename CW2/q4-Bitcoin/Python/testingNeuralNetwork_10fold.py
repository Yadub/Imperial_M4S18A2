import numpy as np
import matplotlib.pyplot as plt
########################################################################

def buildModel(NIL,NHL,NOL):

    # Initialize matrices and arrays to random values
    # np.random.seed(0) # For testing purposes
    W1 = np.random.randn(NHL, NIL) / np.sqrt(NIL)
    c1 = np.zeros((NHL,1))
    W2 = np.random.randn(NOL, NHL) / np.sqrt(NHL)
    c2 = np.zeros((NOL,1))

    return { 'W1': W1, 'c1': c1, 'W2': W2, 'c2': c2}
########################################################################

def trainModel(model, X, y, repeats = 1):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Compute solution and then propogate error back
    for i in range(repeats):
        # Forward propagation
        u1 = W1.dot(X) + c1
        v1 = np.tanh(u1)
        u2 = W2.dot(v1) + c2
        exp_scores = np.exp(u2)
        probs = exp_scores / np.sum(exp_scores)
        # Backpropagation
        delta3 = probs - y
        dW2 = delta3.dot(v1.T)
        dc2 = np.sum(delta3)
        delta2 = (W2.T).dot(delta3) * (1 - np.power(v1, 2))
        dW1 = np.dot(delta2,X.T)
        dc1 = np.sum(delta2)
        # Compute level of error to add
        dW2 += eps2 * W2
        dW1 += eps2 * W1
        # Add Error to the model matrices/arrays
        W1 += -eps1 * dW1
        c1 += -eps1 * dc1
        W2 += -eps1 * dW2
        c2 += -eps1 * dc2
    return { 'W1': W1, 'c1': c1, 'W2': W2, 'c2': c2}
########################################################################

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Forward propagation
    u1 = W1.dot(X) + c1
    v1 = np.tanh(u1)
    u2 = W2.dot(v1) + c2
    exp_scores = np.exp(u2)
    probs = exp_scores / np.sum(exp_scores)
    return probs
########################################################################
def moving_average(x, n) :
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    for i in range(n-1):
        ret[i] /= (i+1)
    ret[n - 1:] /= n
    return ret
# Cite: http://stackoverflow.com/users/110026/jaime
########################################################################

def testResult(model, Xtest, ytest):
# Specefic to this situation. Can be changed for different models
    probs = predict(model, Xtest)
    if np.argmax(probs) == np.argmax(ytest):
        return 1
    else:
        return 0
########################################################################
def trainVals(model, start, num):
    for i in range(start, start+num):
        vals = np.arange(i-Ntau+1,i+1) # indices of prices of interest
        Xt = np.array([price[vals].T, bid[vals].T])
        Xt = Xt.reshape(NIL,1) # Make into a vector
        yt = np.array([priceUp[i], 1 - priceUp[i]]) # Vector of [up, down] booleans
        yt = yt.reshape((2,1))
        # Train the model of this value
        model = trainModel(model, Xt, yt)

    return model

########################################################################
def testVals(model, start, num):
    correct = 0
    for i in range(start, start+num):
        vals = np.arange(i-Ntau+1,i+1)  # indices of prices of interest
        Xt = np.array([price[vals].T, bid[vals].T])
        Xt = Xt.reshape(NIL,1)  # Make into a vector
        yt = np.array([priceUp[i], 1 - priceUp[i]]) # Vector of [up, down] booleans
        yt = yt.reshape((2,1))
        # Compute if the result is correct
        correct += testResult(model, Xt, yt)
    return correct / float(num) # Return % of correct values

########################################################################
if __name__ == '__main__':
    # Load and format the data
    my_data = np.genfromtxt('data.csv', delimiter = ',', dtype = None)

    ID = my_data[1:,0]
    ID = ID.astype(np.float)
    ID = ID[np.argsort(ID)]

    date = my_data[1:,3]
    date = date.astype(np.float)
    date = date[np.argsort(ID)]

    price = my_data[1:,4]
    price = price.astype(np.float)
    price = price[np.argsort(ID)]

    amount = my_data[1:,5]
    amount = amount.astype(np.float)
    amount = amount[np.argsort(ID)]

    bid = my_data[1:,6]
    bid[bid == 'TRUE'] = 1
    bid[bid == 'FALSE'] = 0
    bid = bid.astype(np.float)
    bid = bid[np.argsort(ID)]

    N = len(price) # Set N
    # See if the average price in the next avgSize transactions is higher or lower
    avgSize = 50 # Chosen arbitarily
    movingAvg = moving_average(price, avgSize)
    ND = avgSize
    D = movingAvg[ND:] - movingAvg[:-ND]
    priceUp = np.ones(N-ND)
    priceUp[D < 0] = 0

    Ntau = 100
    dim_X = 2
    NX = Ntau * dim_X

    # Parameters for Neural Network
    eps1 = 1e-6 # learning rate (chosen)
    eps2 = 1e-6 # regularization strength (chosen)
    NIL = NX # input layer size
    NHL = NIL
    NOL = 2 # output layer size

    perfs = np.array([])
    K_fold = 10
    # Set number of values in each fold
    N_samples = (N - Ntau - avgSize) / K_fold
    for K in range(K_fold):
        # Initialize matrices and vectors for the model
        model = buildModel(NIL,NHL,NOL)

        # Train model on 9 of the 10 data sets
        for run in np.random.permutation(K_fold):
            if run != K:
                model = trainVals(model, Ntau + run * N_samples, N_samples)
        # Test on the Kth data set
        performance = testVals(model, Ntau + K * N_samples , N_samples)
        perfs = np.append(perfs, performance)
    print perfs
    print np.average(perfs)

    plt.figure()
    plt.plot(ID,price)
    for K in range(K_fold):
        midval = K*N_samples+ N_samples / 10
        plt.annotate( str.format('{0:.3f}', perfs[K] ) , ( ID[midval], 680))
        plt.plot((ID[K*N_samples], ID[K*N_samples]), (610, 690), 'k-')
    plt.plot((ID[N-1], ID[N-1]), (610, 690), 'k-')
    plt.xlabel('Transaction ID')
    plt.ylabel('Price')
    plt.show()
