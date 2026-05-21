import random
import os
import math
import numpy as np

class ItemRanker:
    
    def __init__(self, n, hidden=64, seed=None, b=None):
        self.rand = random.Random(seed)
        self.w1 = [[self.rand.uniform(-0.1, 0.1) for _ in range(n)] for _ in range(hidden)]
        self.w2 = [self.rand.uniform(-0.1, 0.1) for _ in range(hidden)]
        self.b = b if b!=None else 0.0
    
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    def relu(x):
        return np.maximum(0, x)
        
    def forward(self, x):
      
        self.x = x
        self.h = [] #input_wector
       
        for neuron in self.w1:
           s = sum(w * xi for w, xi in zip(neuron, x))
           self.h.append(self.relu(s))
           
        self.out_raw = sum(w * h_i for w, h_i in zip(self.w2, self.h)) + self.b
        
        if self.out_raw >= 700:
            return 1.0
        if self.out_raw <= -700:
            return 0.0
            
        return np.sigmoid(self.out_raw)
        
    #correct loss on win/loss
    def bce_loss(pred, target):
        pred = max(min(pred, 1 - 1e-9), 1e-9)
        return -(target * math.log(pred) + (1 - target) * math.log(1- pred))
       
    def backward(self, pred, target, lr=0.001):
        d_out = pred - target
        
        # Updated w2 and bias
        for i in range(len(self.w2)):
            self.w2[i] -= lr * d_out * self.h[i]
            
        self.b -= lr * d_out
        
        for i, neuron in enumerate(self.w1):
            if self.h[i] <= 0:
                continue
            
            for j in range(len(neuron)):
                neuron[j] -= lr * d_out * self.w2[i] * self.x[j]
                
                
                
#n = check notes;
model = ItemRanker(n, hidden_size=64)
# build dataset here
x = 1 #match.champion.data
y = 1 if 'win' else 0
dataset = [(x, y)]


for epoch in range(10):
    total_loss = 0
    
    for x, y in dataset:
        pred = model.forward(x)
        loss = model.bce_loss(pred, y)
        model.backward(pred, y, lr=0.001)
        total_loss += loss
      
    print("Epoch", epoch, "Loss:", total_loss)
    