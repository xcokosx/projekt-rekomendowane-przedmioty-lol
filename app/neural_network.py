import random
import os
import math
import numpy as np

class ItemRanker:
    
    def __init__(self, n, hidden=64, seed=None, b=None):
        self.rand = random.Random(seed)
        
        limit = 1 / math.sqrt(n)
        self.w1 = np.array([[self.rand.uniform(-limit, limit) for _ in range(n)] for _ in range(hidden)], dtype=np.float32)
        self.w2 = np.array([self.rand.uniform(-0.1, 0.1) for _ in range(hidden)], dtype=np.float32)
        self.b = 0.0 if b is None else b
        self.b1 = np.zeros(hidden, dtype=np.float32)

    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod # correct win/lose loss
    def bce_loss(pred, target):
        pred = max(min(pred, 1 - 1e-9), 1e-9)
        return -(target * math.log(pred) + (1 - target) * math.log(1 - pred))
        
    def forward(self, x):
        self.x = np.array(x, dtype=np.float32)
        
        s = self.w1 @ self.x + self.b1
        self.h = ItemRanker.relu(s) #input_wector
       
        self.out_raw = float(self.w2 @ self.h + self.b)
        
        if self.out_raw >= 700:
            return 1.0
        if self.out_raw <= -700:
            return 0.0
            
        return ItemRanker.sigmoid(self.out_raw)
        
    def backward(self, pred, target, lr=0.001):
        d_out = np.clip(pred - target, -10, 10) # gradient of loss w.r.t. output
        
        # Updated w2 and bias
        self.w2 -= lr * d_out * self.h
        self.b -= lr * d_out
        
        relu_mask = (self.h > 0).astype(np.float32)
        grad_hidden = d_out * self.w2 * relu_mask

        self.w1 -= lr * grad_hidden[:, None] * self.x[None, :]
        self.b1 -= lr * grad_hidden

        # overfitting regularization
        self.w1 -= lr * 0.0001 * self.w1
        self.w2 -= lr * 0.0001 * self.w2
        self.b1 -= lr * 0.0001 * self.b1

    def save(self, path):
        np.savez(path, w1=self.w1, w2=self.w2, b=self.b, b1=self.b1)

    @staticmethod
    def load(path):
        data = np.load(path)
        model = ItemRanker(
            n=data['w1'].shape[1],
            hidden=data['w1'].shape[0],
            b=data['b']
        )
        model.w1 = data['w1']
        model.w2 = data['w2']
        model.b1 = data['b1']
        return model
