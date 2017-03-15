import numpy as np
import matplotlib.pyplot as plt
########################################################################

def buildModel(NIL,NHL,NOL):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NHL, NIL) / np.sqrt(NIL)
    c1 = np.zeros((NHL,1))
    W2 = np.random.randn(NOL, NHL) / np.sqrt(NHL)
    c2 = np.zeros((NOL,1))

    return { 'W1': W1, 'c1': c1, 'W2': W2, 'c2': c2}
########################################################################

# def forwardPropogate(model, X):
#     W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
#     # Forward propagation
#     u1 = W1.dot(X) + c1
#     v1 = np.tanh(u1)
#     u2 = W2.dot(v1) + c2
#     exp_scores = np.exp(u2)
#     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
# ########################################################################
#
# def backPropogate(model, y, probs):
#     W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
#     # Backpropagation
#     delta3 = probs - y.T
#     # delta3[range(NX), y] -= 1
#     dW2 = (v1.T).dot(delta3)
#     dc2 = np.sum(delta3, axis=0, keepdims=True)
#     delta2 = delta3.dot(W2.T) * (1 - np.power(v1, 2))
#     dW1 = np.dot(X.T, delta2)
#     dc1 = np.sum(delta2, axis=0)
#
#     # Add regularization terms (c1 and c2 don't have regularization terms)
#     dW2 += eps2 * W2
#     dW1 += eps2 * W1
#
#     # Gradient descent parameter update
#     W1 += -eps1 * dW1
#     c1 += -eps1 * dc1
#     W2 += -eps1 * dW2
#     c2 += -eps1 * dc2
#     return { 'W1': W1, 'c1': c1, 'W2': W2, 'c2': c2}
########################################################################

def trainModel(model, X, y, repeats = 100):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Compute solution and then propogate error back
    for i in range(repeats):
        # Forward propagation
        u1 = W1.dot(X) + c1
        v1 = np.tanh(u1)
        u2 = W2.dot(v1) + c2
        exp_scores = np.exp(u2)
        probs = exp_scores / np.sum(exp_scores)
        # print np.shape(probs), np.shape(y)
        # Backpropagation
        delta3 = probs - y
        # print probs
        # print delta3
        # delta3[range(NX), y] -= 1
        # dW2 = (v1.T).dot(delta3)
        dW2 = delta3.dot(v1.T)
        dc2 = np.sum(delta3)
        # delta2 = delta3.dot(W2.T) * (1 - np.power(v1, 2))
        delta2 = (W2.T).dot(delta3) * (1 - np.power(v1, 2))
        # dW1 = np.dot(X.T, delta2)
        dW1 = np.dot(delta2,X.T)
        dc1 = np.sum(delta2, axis=0)

        # Add regularization terms (c1 and c2 don't have regularization terms)
        dW2 += eps2 * W2
        dW1 += eps2 * W1

        # Gradient descent parameter update
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
    return ret[n - 1:] / n
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
        Xt = np.array([price[i], bid[i]])
        for j in range(Ntau-1):
            Xt = np.append(Xt,[price[i-j], bid[i-j]])
        Xt = Xt.reshape((NX,1))
        yt = np.array([priceUp[i], 1 - priceUp[i]]) # Vector of [up, down] booleans
        yt = yt.reshape((2,1))
        model = trainModel(model, Xt, yt)

    return model

########################################################################
def testVals(model, start, num):
    correct = 0
    for i in range(start, start+num):
        Xt = np.array([price[i], bid[i]])
        for j in range(Ntau-1):
            Xt = np.append(Xt,[price[i-j], bid[i-j]])
        Xt = Xt.reshape((NX,1))
        yt = np.array([priceUp[i], 1 - priceUp[i]]) # Vector of [up, down] booleans
        yt = yt.reshape((2,1))
        correct += testResult(model, Xt, yt)
    return correct / float(num)

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

    # Comment out sorting to see the impact
    # plt.plot(ID,price)
    # plt.show()

    # Different definition of if the price went up.
    # 1. Firstly if the next transaction value is higher on lower
    # Not very useful in terms of understanding anything because of financial fluctuations
    N = np.size(price)
    D1 = price[1:N] - price[:N-1] # Level 1 different in price
    priceUp1 = np.ones(N-1)
    priceUp1[D1 < 0] = 0 # I.e. if the price went up or down
    # 2. See if the tranction value a certain number away is higher or lower
    # Still not 100% reasonable due to financial fluctuations
    forecastDepth = 50 # Chosen arbitarily
    D = price[forecastDepth:N] - price[:N-forecastDepth] # Level 1 different in price
    priceUp = np.ones(N-forecastDepth)
    priceUp[D < 0] = 0# I.e. if the price went up or down in
    # 3. See if the average price in the next avgSize transactions is higher or lower
    avgSize = 50 # Chosen arbitarily
    movingAvg = moving_average(price, avgSize)
    D = movingAvg - price[:N-avgSize+1]
    priceUp = np.ones(N-avgSize+1)
    priceUp[D < 0] = 0
    # Plot to visualize
    # plt.plot(ID,price)
    # plt.plot(ID[avgSize-1:],movingAvg,'--')
    # plt.show()

    Ntau = 10
    dim_X = 2
    NX = Ntau * dim_X
    index = Ntau
    Xtrain = np.array([price[index], bid[index]])
    for i in range(Ntau-1):
        Xtrain = np.append(Xtrain,[price[index-i], bid[index-i]])
    Xtrain = Xtrain.reshape((NX,1))

    ytrain = np.array([priceUp[index], 1 - priceUp[index]]) # Vector of [up, down] booleans
    ytrain = ytrain.reshape((2,1))
    # print np.argsort([0,1])
    NX = len(Xtrain) # training set size

    # Parameters for Neural Network
    eps1 = 1e-3 # learning rate (chosen)
    eps2 = 1e-3 # regularization strength (chosen)
    NIL = NX # input layer dimensionality
    NHL = 20
    NOL = 2 # output layer dimensionality

    model = buildModel(NIL,NHL,NOL)
    # model = trainModel(model, Xtrain, ytrain)
    model = trainVals(model, Ntau,10000)
    print testVals(model, Ntau, 8000)
