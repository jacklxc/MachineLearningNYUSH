from matrix import *
import numpy as np
class NeuralNetNumpy:
    def __init__(self, W1, W2, X, y):
        self.X =  X.T
        self.W1 = W1
        self.W2 = W2
        self.y = y.astype(np.int)
        self.yy = self.expand_y(self.y)

    def expand_y(self,y):
        D = np.max(y)+1
        yy = np.zeros((D,y.shape[0]))
        yy[y,np.arange(y.shape[0])] = 1
        return yy

    def affine(self, W, x):
        ones = np.ones((1,x.shape[1]))
        X = np.append(ones, x, axis=0)
        product = np.dot(W,X)
        return product

    def sigmoid(self,mtx):
        return 1/(1+np.exp(-mtx))

    def layer(self, W, x):
        affine_out = self.affine(W,x)
        score = self.sigmoid(affine_out)
        return score

    def forward(self):
        first_layer = self.layer(self.W1, self.X)
        score = self.layer(self.W2, first_layer)
        cache = (first_layer, score)
        return score, cache

    def predict(self, score):
        predictions = np.argmax(score, axis = 0)
        return predictions

    def accuracy(self, predictions):
        return np.mean(predictions == self.y)

    def loss(self,score):
        loss = -np.mean(np.sum(self.yy*np.log(score) + (1-self.yy)*np.log(1-score), axis=0))
        return loss

    def grads(self,cache):
        a, A = cache

        ones = np.ones((1,self.X.shape[1]))
        X = np.append(ones, self.X, axis=0)

        ones = np.ones((1,a.shape[1]))
        a_padding = np.append(ones, a, axis=0)

        dW1 = np.zeros(self.W1.shape)
        dW2 = np.zeros(self.W2.shape)
        for s in range(self.W2.shape[1]):
            aa = np.kron(np.ones((A.shape[0],1)),a_padding[s,:])
            dW2[:,s] = -np.mean(self.yy*(1-A) * aa - (1-self.yy) * A * aa, axis=1)

        for s in range(self.W1.shape[0]):
            aa = np.kron(np.ones((A.shape[0],1)),a[s,:])
            w2 = np.kron(np.ones((A.shape[1],1)),self.W2[:,s+1]).T
            for r in range(self.W1.shape[1]):
                xx = np.kron(np.ones((A.shape[0],1)),X[r,:])
                dW1[s,r] = -np.mean(np.sum(self.yy*(1-A)*aa*w2*(1-aa)*xx - (1-self.yy)*A*aa*w2*(1-aa)*xx,axis=0))
        return dW1, dW2


