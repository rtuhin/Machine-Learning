import numpy as np
import matplotlib.pyplot as plt


def computeCost (X, y, theta):
    m = len(y)    
    j = 0
    theta = theta.reshape(1, 2)
#    print len(X), len(y), len(theta)
    temp = np.subtract(np.dot(theta, X), y)
    j = (temp**2).sum()/(2*m)
    print 'cost:', j
    return j
    
def gradientDescent(X, y, theta, alpha, iterations):
    #m = len(X)
    for i in xrange(iterations):
        temp = np.subtract(np.dot(theta, X), y)
        temp0 = theta[0] - (alpha/float(m))*temp.sum()
        temp1 = theta[1] - (alpha/float(m))*(np.multiply(temp, X)).sum()
        theta[0] = temp0
        theta[1] = temp1
        computeCost(X, y, theta)

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]
m = len(data)
#print m
plt.scatter(X, y)
#print X
X = np.array([np.ones(len(X)), X])
X = X.reshape(2, m)
#print X

theta = np.zeros(2)
#theta = theta.reshape(1, 2)
#print theta

iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)
# run gradient descent
gradientDescent(X, y, theta, alpha, iterations)

y_pred = np.dot(theta, X)
plt.plot(X[1,:], y_pred)

predict1 = np.dot(theta, np.array([1, 3.5]))
print 'For population = 35,000, we predict a profit of', predict1*10000



# print theta to screen
print ('Theta found by gradient descent:', theta[0], theta[1])
