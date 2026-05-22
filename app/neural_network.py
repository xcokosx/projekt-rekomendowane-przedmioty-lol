import random
import os
import math
import numpy as np

class ItemRanker:
    
    def __init__(self, n, num_items, hidden=64, seed=None, b=None):
        self.rand = random.Random(seed)
        
        #hidden
        limit = 1 / math.sqrt(n)
        self.w1 = np.array([[self.rand.uniform(-limit, limit) for _ in range(n)] for _ in range(hidden)], dtype=np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)

        # output - one score per item
        self.w2 = np.array([[self.rand.uniform(-0.1, 0.1) for _ in range(hidden)] for _ in range(num_items)], dtype=np.float32)
        self.b2 = np.zeros(num_items, dtype=np.float32)

        self.num_items = num_items

    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod # correct win/lose loss
    def bce_loss(pred, target):
        pred = np.clip(pred, 1e-9, 1 - 1e-9)
        return -np.sum(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        
    def forward(self, x):
        self.x = np.array(x, dtype=np.float32)
        
        s = self.w1 @ self.x + self.b1
        self.h = ItemRanker.relu(s) #input_wector
       
        logits = self.w2 @ self.h + self.b2
        self.out_raw = ItemRanker.sigmoid(logits)
        
        return self.out_raw
        
    def backward(self, pred, target, lr=0.001):
        d_out = pred - target  # gradient of loss w.r.t. output
        
        # update w2 and bias
        self.w2 -= lr * (d_out[:, None] * self.h[None, :])
        self.b2 -= lr * d_out
        
        # backprop hidden
        relu_mask = (self.h > 0).astype(np.float32)
        grad_hidden = (d_out @ self.w2) * relu_mask # [hidden,]

        # update hidden
        self.w1 -= lr * grad_hidden[:, None] * self.x[None, :]
        self.b1 -= lr * grad_hidden

        # overfitting regularization
        self.w1 -= lr * 0.0001 * self.w1
        self.w2 -= lr * 0.0001 * self.w2
        self.b1 -= lr * 0.0001 * self.b1

    def save(self, path):
        np.savez(path, w1=self.w1, w2=self.w2, b1=self.b1, b2=self.b2)

    @staticmethod
    def load(path):
        data = np.load(path)
        
        model = ItemRanker(n=data['w1'].shape[1], hidden=data['w1'].shape[0], num_items=data['w2'].shape[0])

        model.w1 = data['w1']
        model.w2 = data['w2']
        model.b1 = data['b1']
        model.b2 = data['b2']
        return model
