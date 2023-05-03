import numpy as np
import random as rd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['figure.autolayout'] = True

fig = plt.figure(figsize=(10, 7))
fig.tight_layout()


def sigmoid(x, derv=False):
    f = 1.0/(1 + np.exp(-x))
    if derv:
        return f*(1 - f)
    return f


class NNet:

    rand = lambda self: rd.random()

    def __init__(self, inputs=10, outputs=3, epochs=200):
        self.inputs = inputs
        self.outputs = outputs
        self.epochs = epochs

    def __call__(self, inputs, outputs, I, J):
        error_set = 1
        while error_set >= 0.001:
            for i in self.axis:
                Z = self.weights[i]
                mi, ni = Z.shape
                xi, yi = np.meshgrid(range(ni), range(mi))
                self.plots[i].cla()
                self.plots[i].plot_surface(xi, yi, Z, color='black')
                
                
                if i == self.axis[0]:
                    self.layers[i] = self.weights[i].T.dot(inputs)
                    self.slayers[i] = sigmoid(self.layers[i])
                    
                else:
                    self.layers[i] = self.weights[i].T.dot(self.slayers[i+1])
                    self.slayers[i] = sigmoid(self.layers[i])

            error = 0
            delta = 0
            delta_store = {}
            for i in self.baxis:
                if i == self.baxis[0]:
                    error = pow(outputs - self.slayers[i], 2)
                    error_set = np.sum(error)
                    print(error_set)
                    delta = 2*(outputs - self.slayers[i])*sigmoid(self.layers[i], derv=True)
                    #self.weights[i] += delta.T
                else:
                    error = self.weights[i-1].dot(delta)
                    delta = error*sigmoid(self.slayers[i], derv=True)
                    #self.weights[i] += delta.T
                delta_store[i] = delta
                
            for i in self.baxis:
                self.weights[i] += delta_store[i].T

        kk = 1
        for i in self.axis:
            Z = self.weights[i]
            mi, ni = Z.shape
            xi, yi = np.meshgrid(range(ni), range(mi))
            self.plots[i].cla()
            self.plots[i].set_title(f'Layer: {kk} | Left: {J - I}')
            self.plots[i].plot_surface(xi, yi, Z, cmap='hsv')
            kk += 1
        plt.pause(0.00000001)

    def buildWeights(self):
        i = self.inputs
        j = self.outputs
        self.weights = {}
        self.layers = {}
        self.slayers = {}
        self.plots = {}
        self.axis = list(range(i, j, -1))
        self.baxis = self.axis[::-1]
        ii = 1
        for k in self.axis:
            self.weights[k] = np.array([[self.rand() for n in range(k-1)] for m in range(k)])
            self.layers[k] = np.array([0 for m in range(k-1)])
            self.slayers[k] = np.array([0 for m in range(k-1)])
            self.plots[k] = fig.add_subplot(3, 3, ii, projection='3d')
            ii += 1

    def testWeights(self, inputs, outputs):
        for i in self.axis:
            if i == self.axis[0]:
                self.layers[i] = self.weights[i].T.dot(inputs)
                self.slayers[i] = sigmoid(self.layers[i])
            else:
                self.layers[i] = self.weights[i].T.dot(self.slayers[i+1])
                self.slayers[i] = sigmoid(self.layers[i])
                
        norm_outputs = self.unnorm(self.slayers[self.axis[-1]], outputs)
        

    def normalize(self, x):
        min_, max_ = min(x), max(x)
        norm = (x - min_)/(max_ - min_)
        return norm

    def unnorm(self, x, y):
        min_, max_ = min(x), max(x)
        un = y*(max_ - min_) + min_
        return un

N = 400
INPUTS = [[rd.randint(1, 100)/100 for k in range(10)] for m in range(N)]
OUTPUTS = [[rd.randint(1, 100)/100 for k in range(4)] for m in range(N)]

nnet = NNet(inputs=len(INPUTS[0]), outputs=len(OUTPUTS[0]))
nnet.buildWeights()

for ii, (inputs, outputs) in enumerate(zip(INPUTS, OUTPUTS)):

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    norm_inputs = nnet.normalize(inputs)
    norm_outputs = nnet.normalize(outputs)

    nnet(norm_inputs, norm_outputs, ii, N)



plt.show()
