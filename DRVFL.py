import numpy as np
import scipy.special as sp
from math import sqrt

from sklearn import metrics
import sklearn.datasets as skd
import sklearn.model_selection as skm
import sklearn.preprocessing as skp

from dataclasses import asdict, astuple, dataclass, field, replace

std = skp.MinMaxScaler(feature_range=(0,1))

@dataclass
class Model():
    Weight: list[np.ndarray] | None = field(default_factory=list)
    Biases: list[np.ndarray] | None = field(default_factory=list)
    Beta: np.ndarray | None = field(default_factory=list)
    n_Layer: int | None = None
    
    def __post_init__(self):
        self.Weight = [[]] * self.n_Layer
        self.Biases = [[]] * self.n_Layer
    
class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return sp.expit(x)
    
    @staticmethod
    def brelu(x):
        return np.where(x <= 0, 0, np.where(x > 1, 1, x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        return sp.softmax(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def binary(x):
        return np.where(x <= 0, -1, 1)
    
class DeepRVFL:
    def __init__(self, n_nodes: list[int], lmbdas, act_func, task):
        task_list = ['regression', 'classification']
        acfn_list = ['sigmoid', 'brelu', 'tanh', 'softmax', 'relu', 'binary']
        
        assert task.lower() in task_list, "Task Type Not Defined!"
        assert act_func.lower() in acfn_list, "Acivation Function Not Found!"
        
        self.task = task
        self.n_nodes = n_nodes
        self.lmbdas = lmbdas
        self.act_func = getattr(ActivationFunction(), act_func)
        self.model = Model(n_Layer=len(self.n_nodes))
        
    def train(self, X1, T1):
        self.D = None
        self.Hi = list()
        self.I = X1.copy()
        
        for idx in range(len(self.n_nodes)):
            n_feature = self.I.shape[1]
            W, b = self.generateNodes(self.lmbdas, n_feature, self.n_nodes[idx])
            
            self.I = self.act_func(self.I @ W + b)
            
            if len(self.Hi) == 0:
                self.Hi = self.I
            else:
                self.Hi = self.combineH(self.Hi, self.I)

            self.model.Weight[idx] = W
            self.model.Biases[idx] = b
            
        self.D = self.combineD(self.Hi, X1)
        self.model.Beta = self.calcBeta(T1, self.D)
        
        resErr, Y1 = self.calcYresult(self.model.Beta, T1, self.D)
        score = self.calcRMSE(resErr, X1.shape[0])
        
        if self.task == 'classification':
            acc = self.calcAccuracy(T1, Y1)
            return {"RMSE" : score, "Accuracy" : acc}
        return {"RMSE" : score}
    
    def predict(self, X2, T2):
        Hi2 = list()
        I2 = X2.copy()
        n_sample, n_feature = X2.shape
        
        for idx in range(len(self.n_nodes)):
            I2 = self.act_func(I2 @ self.model.Weight[idx] + self.model.Biases[idx])
            
            if len(Hi2) == 0:
                Hi2 = I2
            else:
                Hi2 = self.combineH(Hi2, I2)
                
        D2 = self.combineD(Hi2, X2)
        resErr2, Y2 = self.calcYresult(self.model.Beta, T2, D2)
        score = self.calcRMSE(resErr2, n_sample)
        
        if self.task == "classification":
            acc = self.calcAccuracy(T2, Y2)
            return {"RMSE": score, "Accuracy": acc}
        return {"RMSE": score}
            
    def generateNodes(self, lmbda, n_feature, n_nodes):
        W = lmbda * (2 * np.random.rand(n_feature, n_nodes) - 1)
        b = lmbda * (2 * np.random.rand(1, n_nodes) - 1)
        return W, b
    
    @staticmethod
    def combineD(H, X):
        return np.concatenate([np.ones_like(X[:, 0:1]), H, X], axis=1)
    
    @staticmethod
    def combineH(Hi, I):
        return np.concatenate([Hi, I], axis=1)
    
    def calcBeta(self, T, D):
        return np.linalg.pinv(D) @ T
    
    @staticmethod
    def calcYresult(Beta, T, D):
        Y = D @ Beta
        resErr = Y - T
        return resErr, Y
    
    @staticmethod
    def calcRMSE(resErr, n_sample):
        return sqrt(np.sum(np.sum(resErr ** 2, axis=0) / n_sample, axis=0))
    
    @staticmethod
    def calcAccuracy(T, Y):
        Y = np.argmax(Y, axis=1)
        T = np.argmax(T, axis=1)
        return metrics.accuracy_score(T, Y)
    
if __name__ == "__main__":
    def load_mnist():
        import tensorflow as tf
        from keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        from sklearn.model_selection import train_test_split
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Convert to 1D
        x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

        # Normalize X to between -1 and 1
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        scaler.fit(x_test)
        x_test = scaler.transform(x_test)

        # Convert the Target(y) to One Hot Encoded
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        y_train = to_categorical(y_train, 10).astype(int)
        y_test = to_categorical(y_test, 10).astype(int)

        X2, X1, T2, T1 = train_test_split(x_train, y_train, test_size=0.83333)
        
        return X2, X1, T2, T1
    
    X2, X1, T2, T1 = load_mnist()
    drvfl = DeepRVFL([6000,20], 1, 'sigmoid', 'classification')
    train_res = drvfl.train(X1, T1)
    valid_res = drvfl.predict(X2, T2)
    
    print(train_res)
    print(valid_res)