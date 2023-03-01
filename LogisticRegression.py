class LogRegression():
    def __init__(self, lr=0.001, n_iters=100):
        self.n_iters = n_iters
        self.lr = lr
        self.weights = None
        self.bias = None

    # fit and train model
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # begin iteration
        for _ in range(self.n_iters):
            # make linear prediction
            linear_pred = np.dot(X, self.weights) + self.bias

            # flatten output to be between 0 and 1
            predictions = sigmoid(linear_pred)

            # calculate gradient and error
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # update weights and bias respectively
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # function to make predictions

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)

        # will make it so anything less than .5 is made to be 0 and anything greater than .5 to be 1
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]

        return class_pred

    # calculate accuracy of model


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


# function to constrain output values to between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


log = LogRegression(n_iters=1000)

log.fit(x_train, y_train)

y_pred = log.predict(x_test)

acc = accuracy(y_pred, y_test)

df = pd.DataFrame(y_test, y_pred)