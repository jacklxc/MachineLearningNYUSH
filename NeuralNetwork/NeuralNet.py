from matrix import *

class NeuralNet:
    def __init__(self, W1, W2, X, y):
        self.X =  transpose(X)
        self.W1 = W1
        self.W2 = W2
        self.y = y # list

    def affine(self, W, x):
        X = x.padding()
        product = dot(W,X)
        return product

    def layer(self, W, x):
        affine_out = self.affine(W,x)
        score = sigmoid(affine_out)
        return score

    def forward(self):
        first_layer = self.layer(self.W1, self.X)
        score = self.layer(self.W2, first_layer)
        return score

    def predict(self, score):
        predictions = argmax(score, axis = 1)
        return predictions

    def accuracy(self, predictions):
        total = len(self.y)
        correct = 0
        for i in range(total):
            if predictions[i]==self.y[i]:
                correct+=1
        return correct/total

    def loss(self, score):
        self.yy = zeros(score.shape)
        for i in range(score.shape[1]):
            self.yy[int(self.y[i]),i] = 1
        loss = -mean(self.yy * log(score) + (-self.yy + 1) * log(-score+1)) * score.shape[0]
        return loss
