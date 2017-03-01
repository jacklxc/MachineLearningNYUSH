import math
import numpy as np

def Mean(numbers):
    """
    Self-implemented mean function.
    Takes a list of numbers and compute the mean.
    """
    mean = 0
    for number in numbers:
        mean += number
    mean /= len(numbers)
    return mean

def Std(numbers):
    """
    Self-implemented std function.

    Takes a list of numbers and returns 
    the standard deviation of the numbers.
    std = sqrt(E(x^2)-(E(x)^2))
    """
    square = []
    for number in numbers:
        square.append(number*number)
    square_mean = Mean(square)
    mean = Mean(numbers)
    mean_sq = mean*mean
    std = math.sqrt(square_mean - mean_sq)
    return std

def normalize(numbers):
    mean = Mean(numbers)
    std = Std(numbers)
    normalized = []
    for number in numbers:
        normalized.append((number - mean)/std)
    return normalized, mean, std

def normalize_file(read_name, write_name):
    with open(read_name,"r") as f:
        lines = f.readlines()

    x1 = []
    x2 = []
    y = []

    for line in lines:
        numbers = line.split(",")
        x1.append(int(numbers[0]))
        x2.append(int(numbers[1]))
        y.append(int(numbers[2]))

    normalized_x1, mean_x1, std_x1 = normalize(x1)
    normalized_x2, mean_x2, std_x2 = normalize(x2)
    normalized_y, mean_y, std_y = normalize(y)

    write_list = []
    for i in range(len(x1)):
        line = [normalized_x1[i],normalized_x2[i],normalized_y[i]]
        write_list.append(line)
    # The last 2 lines are mean and std
    line = [mean_x1, mean_x2, mean_y]
    write_list.append(line)
    line = [std_x1, std_x2, std_y]
    write_list.append(line)

    with open(write_name,"w") as f:
        for item in write_list:
            f.write(','.join(map(str, item))+"\n")

#--------------------------------------------------------------------#
# All functions below uses numpy

def load_data(file_name):
    """
    Returns:
    - x: numpy matrix in shape (M,N).
    - y: numpy array in shape (N).
    - mean: numpy array, mean of (x1,x2,...,xn,y).
    - std: numpy array, std of (x1,x2,...,xn,y).
    """
    with open(file_name,"r") as f:
        lines = f.readlines()
    numbers = []
    for line in lines:
        splited = line.split(",")
        splited_int = []
        for item in splited:
            splited_int.append(float(item))
        numbers.append(splited_int)
    # Pop out std and mean at the lend of data
    std = np.array(numbers.pop())
    mean = np.array(numbers.pop())

    numbers = np.array(numbers)
    x = numbers[:,:-1].T # Remove the dimension for y
    y = numbers[:,-1]

    return x, y, mean, std


def f(w,x):
    """
    Predicts y.
    Inputs:
    - w: numpy array in shape (M+1).
    - x: numpy matrix in shape (M,N).
    Returns:
    - f: numpy array of predictions in shape (N).
    """
    M, N = x.shape
    f = np.ones((N)) * w[0]
    for i in range(N):
        f[i] += np.sum(w[1:]*x[:,i])
    return f

def gradient(w,x,y):
    """
    Inputs:
    - w: numpy array in shape (M+1).
    - x: numpy matrix in shape (M,N).
    - y: numpy array in shape (N).
    """
    M, N = x.shape
    dws = np.zeros((M+1))
    dws[0] = np.mean(f(w,x) - y)
    for i in range(M):
        dws[i+1] = np.mean((f(w,x) - y) * x[i,:])
    return dws


def gradient_descent(w,x,y,alpha):
    """
    Initialize w to all 0 and do gradient descent.
    Inputs:
    - w: numpy array in shape (M+1).
    - x: numpy matrix in shape (M,N).
    - y: numpy array in shape (N).
    - alpha: leanring rate.
    """
    return w-alpha*gradient(w,x,y)

def loss(w,x,y):
    return np.mean(np.square(f(w,x) - y))/2

def train(x,y,alpha, max_iter=80, verbose = False):
    M,N = x.shape
    w = np.zeros((M+1))
    for iteration in range(max_iter):
        w = gradient_descent(w,x,y,alpha)
        if verbose:
            print(loss(w,x,y))
    return loss(w,x,y), w

def predict(w,x,mean,std):
    """
    Only supports single prediction.
    """
    x_norm = (x - mean[:-1])/std[:-1]
    y_norm = f(w,x_norm)[0]
    y = y_norm * std[-1] + mean[-1]
    return y

def sgd(w,x,y,alpha):
    M, N = x.shape
    for n in range(N):
        dws = np.zeros((M+1))
        dws[0] = f(w,np.expand_dims(x[:,0],axis=1))[0] - y[0]
        for m in range(M):
            dws[m+1] = (f(w,np.expand_dims(x[:,n],axis=1))[0] - y[n])* x[m,n]
        w = w - alpha * dws
    return w

def train_sgd(x,y,alpha, max_iter=3, verbose = False):
    M,N = x.shape
    w = np.zeros((M+1))
    for iteration in range(max_iter):
        w = sgd(w,x,y,alpha)
        if verbose:
            print(loss(w,x,y))
        x,y = shuffle(x,y)
    return w

def shuffle(x,y):
    """
    Inputs:
    - x: numpy matrix in shape (M,N).
    - y: numpy array in shape (N).
    """
    _,N = x.shape
    indices = np.arange(N)
    np.random.shuffle(indices)
    new_x = x[:,indices]
    new_y = y[indices]
    return new_x, new_y
