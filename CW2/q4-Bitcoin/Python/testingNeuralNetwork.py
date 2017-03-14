import numpy as np
import matplotlib.pyplot as plt
########################################################################

def buildModel(NIL,NHL,NOL):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(NIL, NHL) / np.sqrt(NIL)
    c1 = np.zeros((1, NHL))
    W2 = np.random.randn(NHL, NOL) / np.sqrt(NHL)
    c2 = np.zeros((1, NOL))

    return { 'W1': W1, 'c1': c1, 'W2': W2, 'c2': c2}
########################################################################

def forwardPropogate(model, X):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Forward propagation
    u1 = X.dot(W1) + c1
    v1 = np.tanh(u1)
    u2 = v1.dot(W2) + c2
    exp_scores = np.exp(u2)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
########################################################################

def backPropogate(model, y, probs):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Backpropagation
    delta3 = probs
    delta3[range(NX), y] -= 1
    dW2 = (v1.T).dot(delta3)
    dc2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(v1, 2))
    dW1 = np.dot(X.T, delta2)
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

def trainModel(model, X, y, repeats = 100):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Compute solution and then propogate error back
    for i in range(repeats):
        probs = forwardPropogate(model, X)
        model = backPropogate(model, y, probs)
    return model
########################################################################

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, c1, W2, c2 = model['W1'], model['c1'], model['W2'], model['c2']
    # Forward propagation
    u1 = X.dot(W1) + c1
    v1 = np.tanh(u1)
    u2 = v1.dot(W2) + c2
    exp_scores = np.exp(u2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
########################################################################
def moving_average(x, n) :
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
# Cite: http://stackoverflow.com/users/110026/jaime
########################################################################


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
priceUp1 = D1[D1 < 0] # I.e. if the price went up or down
# 2. See if the tranction value a certain number away is higher or lower
# Still not 100% reasonable due to financial fluctuations
forecastDepth = 50 # Chosen arbitarily
D = price[forecastDepth:N] - price[:N-forecastDepth] # Level 1 different in price
priceUp = np.ones(N-forecastDepth)
priceUp = D[D < 0] # I.e. if the price went up or down in
# 3. See if the average price in the next avgSize transactions is higher or lower
avgSize = 50 # Chosen arbitarily
movingAvg = moving_average(price, avgSize)

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

ytrain = np.array([price[index], bid[index]])
ytrain = ytrain.reshape((2,1))
print np.shape(Xtrain)
NX = len(Xtrain) # training set size

# Parameters for Neural Network
eps1 = 1e-3 # learning rate (chosen)
eps2 = 0.01 # regularization strength (chosen)
NIL = NX # input layer dimensionality
NHL = 10
NOL = 2 # output layer dimensionality

W1 = np.random.randn(NHL, NIL) / np.sqrt(NIL)
print np.size(W1.dot(Xtrain)

# model = buildModel(NIL,NHL,NOL)
# model = trainModel(model, Xtrain, ytrain)
