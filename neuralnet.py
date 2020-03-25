import numpy as np 
import math
import sys

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def sgd(data, test, epochs, hiddenUnits, initFlag, learningRate):
    inputs = data.splitlines()
    inputSize = len(inputs[0].split(",")) - 1
    outputSize = 10
    if initFlag == 1: #random weights
        alpha = np.random.uniform(-0.1, 0.1, (hiddenUnits, inputSize + 1))
        beta = np.random.uniform(-0.1, 0.1, (outputSize, hiddenUnits + 1))
        #bias initialize
        for i in range(len(alpha)):
            alpha[i][0] = 0 
        for i in range(len(beta)):
            beta[i][0] = 0 
    else: #zero weights
        alpha = np.zeros((hiddenUnits, inputSize + 1))
        beta = np.zeros((outputSize, hiddenUnits + 1))
    trainCrossEntropy = []
    testCrossEntropy = []

    for e in range(epochs):
        crossEntropy = 0
        for point in inputs:
            p = point.split(",")
            y = np.zeros((1, 10))
            y[0][int(p[0])] = 1 #one-hot encoding 
            y = y.T
            x = np.array([[1] + list(map(lambda x: int(x), p[1:]))]).T #bias init x0 as 1
            (a, b, z, yHat) = NNForward(x, y, alpha, beta)
            gradientAlpha, gradientBeta = NNBackward(x, y, alpha, beta, a, b, z, yHat)
            alpha = alpha - learningRate*gradientAlpha
            beta = beta - learningRate*gradientBeta
        trainCrossEntropy.append(findCrossEntropy(data, alpha, beta))
        testCrossEntropy.append(findCrossEntropy(test, alpha, beta))
    return alpha, beta, trainCrossEntropy, testCrossEntropy

def findCrossEntropy(data, alpha, beta):
    ce = 0
    inputs = data.splitlines()
    for d in inputs:
        p = d.split(",")
        y = np.zeros((1, 10))
        y[0][int(p[0])] = 1 #one-hot encoding 
        y = y.T
        x = np.array([[1] + list(map(lambda x: int(x), p[1:]))]).T #bias init x0 as 1
        (_ ,_ ,_ , yHat) = NNForward(x, y, alpha, beta)
        ce += y.T.dot(np.log(yHat))
    return -1/len(inputs)*ce[0][0]

def linearForward(x, alpha):
    return alpha.dot(x)

def sigmoidForward(a):
    s = np.vectorize(lambda x: 1/(1 + math.e ** (-x)))
    return s(a)

def softmaxForward(b):
    denom = np.sum(np.exp(b))
    sm = np.vectorize(lambda x: math.e**x/denom)
    return sm(b) 

def NNForward(x, y, alpha, beta):
    a = linearForward(x, alpha)
    z = sigmoidForward(a)
    z = np.insert(z, 0, 1, axis=0)
    b = linearForward(z, beta)
    yHat = softmaxForward(b)
    return a, b, z, yHat

def linearBackward(a, b, gb):
    galpha = gb.dot(a.T)
    ga = b.T.dot(gb)
    return galpha, ga

def sigmoidBackward(z, gz):
    return np.multiply(gz, np.multiply(z, 1 - z))

def NNBackward(x, y, alpha, beta, a, b, z, yHat):
    gb = yHat - y
    gbeta, gz = linearBackward(z, beta[:, 1:], gb)
    z = z[1:]
    ga = sigmoidBackward(z, gz)
    galpha, gx = linearBackward(x, a, ga)
    return galpha, gbeta 


def predict(data, out, alpha, beta):
    res = ""
    correct = 0
    inputs = data.splitlines()
    for line in inputs:
        p = line.split(",")
        y = np.zeros((1, 10))
        y[0][int(p[0])] = 1 #one-hot encoding 
        y = y.T
        x = np.array([[1] + list(map(lambda x: int(x), p[1:]))]).T
        (_, _, _, yHat) = NNForward(x, y, alpha, beta)
        predicted = str(np.argmax(yHat))
        if predicted == p[0]:
            correct += 1
        res += "%s\n" % predicted
    writeFile(out, res)
    return 1 - correct/len(inputs)



def writeMetric(metricsOut, trainError, trainCrossEntropy, testError, testCrossEntropy):
    res = ""
    assert(len(trainCrossEntropy) == len(testCrossEntropy))
    for epoch in range(len(trainCrossEntropy)):
        res += "epoch=%d crossentropy(train): %f\n" % (epoch + 1, 
            trainCrossEntropy[epoch])
        res += "epoch=%d crossentropy(test): %f\n" % (epoch + 1, 
            testCrossEntropy[epoch])
    res += "error(train): %f\nerror(test): %f" % (trainError, testError)
    writeFile(metricsOut, res)

if __name__ == "__main__":
    
    trainInput = sys.argv[1] 
    testInput = sys.argv[2]
    trainOut = sys.argv[3]
    testOut = sys.argv[4]
    metricsOut = sys.argv[5]
    epochs = int(sys.argv[6])
    hiddenUnits = int(sys.argv[7])
    initFlag = int(sys.argv[8])
    learningRate = float(sys.argv[9])

    trainData = readFile(trainInput)
    testData = readFile(testInput)

    #crossentropy will be an array of CE in order of epochs
    (alpha, beta, trainCrossEntropy, testCrossEntropy) = sgd(trainData, 
        testData, epochs, hiddenUnits, initFlag, learningRate)

    trainError = predict(trainData, trainOut, alpha, beta)
    testError = predict(testData, testOut, alpha, beta)

    writeMetric(metricsOut, trainError, trainCrossEntropy, testError, 
        testCrossEntropy)