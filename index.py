import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


class perceptronSimple:
    def __init__(self, n_inputs, learningRate):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learningRate

    def predict(self, X):
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i] + self.b)
            if(y_est[i] >= 0):
                y_est[i] = 1
            else:
                y_est[i] = 0
        return y_est

    def fit(self, X, Y, epochs=30):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1, 1))
                self.w += self.eta * (Y[:,i] - y_est) * X[:, i]
                self.b += self.eta * (Y[:,i] - y_est)

    def drawPerceptron2d(self, model):
        w1, w2, b = model.w[0], model.w[1], model.b
        plt.plot([-2, 2], [(1/w2) * (-w1*(-2)-b), (1/w2)*(-w1*(2)-b)])

def getDataOfColumnsCsv(df,column):
    lista = []
    for i in range(len(df)):
        aux = df[i]
        lista.append(aux[column])
    lista = np.asarray(lista)
    return lista.reshape(-1,1)

if __name__ == "__main__":
    # Parámetros
    learning_rate = 0.1
    max_epochs = 1000
    df = pd.read_csv('spheres2d70.csv',header=0)
    test = df.values
    aux = []
    x1 = getDataOfColumnsCsv(test,0)
    x2 = getDataOfColumnsCsv(test,1)
    x3 = getDataOfColumnsCsv(test,2)
    Y = getDataOfColumnsCsv(test,3)
    # X_test = getDataOfColumnsCsv(test,2)

    X = np.concatenate((x1.T,x2.T,x3.T),axis=0)
    Y = Y.reshape(1,-1)
    print(X.shape, Y.shape)

    model = perceptronSimple(3, 0.5)
    model.fit(X, Y,1000)
    # print(model.predict(X))

    # dibujo
    p = X.shape[1]
    for i in range(p):
        if(Y[:,i] <= 0):
            plt.plot(X[0, i], X[1, i], 'or')
        else:
            plt.plot(X[0, i], X[1, i], 'og')


    plt.title('test perceptron')
    plt.grid('on')
    # plt.xlim([-2, 2])
    # plt.ylim([-2, 2])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    model.drawPerceptron2d(model)
    plt.show()
