# neural-net
Python implementation of a one hidden layer neural network trained on .csv files using stochastic gradient descent.

This takes in the following commandline arguments:
1. path to the training dataset
2. path to the test dataset
3. path to write predicted output based on training data
4. path to write predicted output based on test data
5. path to write test and training error and loss per epoch 
6. number of epochs to train
7. number of hidden units present in the hidden layer
8. "1" to initialize all weight parameters randomly between -0.1 and 0.1, "2" to initialize all parameters as 0
9. learning rate.
