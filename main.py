import numpy as np
from PIL import ImageTk, Image

def weights(j, W, T, X, S):
    W[:,j] = (S * X)/(S - 1 + np.sum(X))
    T[:,j] = X

def outputValue(j: int, W: np.ndarray, X: np.ndarray):
    """
    Рассчитывается значение выхода нейрона с учетом коэффициентов кратковременной памяти:
        Параметры:
            j - номер кластера
            W - синапсы кратковременной памяти
            X - входной образ
    """
    return (W[:,j] * X).sum()

def resemblanceValue(j: int, T: np.ndarray, X: np.ndarray):
    """
    Рассчитывается количественная мера сходства входного образа X с кластером:
        Параметры:
            j - номер кластера
            T - синапсы долговременной памяти
            X - входной образ
    """
    return (T[:,j] * X).sum()/X.sum()

def adaptation(j: int, X, W: np.ndarray, T: np.ndarray, v: float, S: float):
    """
    Пересчёт весовых коэффициентов:
        Параметры:
            j - номер кластера
            W - синапсы кратковременной памяти
            T - синапсы долговременной памяти
            v - коэффициент скорости адаптации
            S - определяет степень влияния 
                входного образа на кратковременную память W
    """
    W[:,j] = (1 - v)*W[:,j] + v*(S * X)/(S - 1 + np.sum(X))
    T[:,j] = (1 - v)*T[:,j] + v*X

def imgConvert(filename: str):
    """
    Преобразует файл filename в массив нампай из единиц и нулей:
        Параметры:
            filename - название файла
    """
    img = Image.open(filename)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.asarray(img.convert('L'))
    X = 1 - (img_array.reshape(784)/255.0)
    return X

def giveResult(X):
    Result = network.recognition(X, 28*28, 10)
    Result = 100*np.around(Result, decimals = 4)
    return Result

class NeuralNetworkART:
    def __init__(self, X, n = 28*28, m = 10, Rk = 0.7, S = 2):
        """
        Инициализация нейронной сети адаптивного резонанса:
            Параметры:
                n - количество "входов"
                m - количество "выходов"(максимальное количество кластеров)
                Rk - параметр сходства
                S - определяет степень влияния 
                    входного образа на кратковременную память W
        """
        self.S = S
        self.Y = np.zeros(m)
        self.W = np.zeros((n, m))
        self.T = np.zeros((n, m))
        self.R = np.zeros(m)
        self.Rk = Rk
        self.active = 0
        # Создание первого кластера входного образа X
        weights(self.active, self.W, self.T, X, self.S)
    def recognition(self, X, n = 28*28, m = 10):
        """
        Создание новых кластеров или распознавание принимаемых образов:
            Параметры:
                X - входной образ
        """
        for j in range(0, self.active + 1):
            self.Y[j] = outputValue(j, self.W, X)
        if np.max(self.Y) == 0 and self.active != m - 1:
            self.active += 1
            weights(self.active, self.W, self.T, X, self.S)
        else:
            for j in range(0, self.active + 1):
                self.R[j] = resemblanceValue(j, self.T, X)
            if np.max(self.R) < self.Rk and self.active != m - 1:
                self.active += 1
                weights(self.active, self.W, self.T, X, self.S)
            else:
                adaptation(np.argmax(self.R), X, self.W, self.T, 0.1, self.S)
        return self.R
        
filename = "0.png"
X = imgConvert(filename)
network = NeuralNetworkART(X, 28*28, 10, 0.7, 2)

for i in range (1, 10):
    filename = ("{}.png".format(i))
    X = imgConvert(filename)
    network.recognition(X, 28*28, 10)
