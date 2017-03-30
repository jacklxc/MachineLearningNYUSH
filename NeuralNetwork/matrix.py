import math
import copy
import numpy as np
"""
Module for operating 2-d matrices by mimicing Numpy (but not very similar).
Only necessary functions are written.
"""

class Matrix:
    def __init__(self, size = (1), initial_number = 0, MATRIX = None):
        """
        MATRIX is a nested list. 
        """
        if MATRIX:
            self.matrix = MATRIX
            self.shape = self._shape()
        else:
            for dim in range(len(size)-1,-1,-1):
                if dim == len(size)-1:
                    matrix = []
                    for i in range(size[dim]):
                        matrix.append(initial_number)
                else:
                    new_matrix = []
                    for i in range(size[dim]):
                        new_matrix.append(copy.deepcopy(matrix))
                    matrix = new_matrix
            self.matrix = matrix
            self.shape = size

    def __getitem__(self,key):
        return self.matrix[key[0]][key[1]]

    def __setitem__(self,key,value):
        self.matrix[key[0]][key[1]] = value

    def __repr__(self):
        return repr(self.matrix)

    def __add__(self,mtx2):
        if isinstance(mtx2,Matrix):
            D0, D1 = self.shape
            D00, D11 = mtx2.shape
            if D0!=D00 or D1!=D11:
                raise ValueError("Shapes not aligned.")

            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] + mtx2[i,j]
        else:
            D0, D1 = self.shape
            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] + mtx2
        return mtx

    def __sub__(self,mtx2):
        if isinstance(mtx2,Matrix):
            D0, D1 = self.shape
            D00, D11 = mtx2.shape
            if D0!=D00 or D1!=D11:
                raise ValueError("Shapes not aligned.")

            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] - mtx2[i,j]
        else:
            D0, D1 = self.shape
            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] - mtx2
        return mtx

    def __mul__(self,mtx2):
        if isinstance(mtx2,Matrix):
            D0, D1 = self.shape
            D00, D11 = mtx2.shape
            if D0!=D00 or D1!=D11:
                raise ValueError("Shapes not aligned.")

            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] * mtx2[i,j]
        else:
            D0, D1 = self.shape
            mtx = zeros(self.shape)
            for i in range(D0):
                for j in range(D1):
                    mtx[i,j] = self[i,j] * mtx2
        return mtx

    def __neg__(self):
        D0, D1 = self.shape
        mtx = zeros(self.shape)
        for i in range(D0):
            for j in range(D1):
                mtx[i,j] = -self[i,j]
        return mtx

    def _shape(self):
        matrix = self.matrix
        done = False
        shape = []
        while not done:
            done = not isinstance(matrix[0], list)
            shape.append(len(matrix))
            matrix = matrix[0]
        return tuple(shape)

    def padding(self, number = 1, axis = 0):
        D0, D1 = self.shape
        matrix = copy.deepcopy(self.matrix)
        if axis == 0:
            numbers = []
            for i in range(D1):
                numbers.append(number)
            matrix.insert(0,numbers)
        else:
            for i in range(D0):
                matrix[i].insert(0,number)
        mtx = Matrix(MATRIX = matrix)
        return mtx

def zeros(size):
    mtx = Matrix(size = size, initial_number = 0)
    return mtx

def ones(size):
    mtx = Matrix(size = size, initial_number = 1)
    return mtx

def dot(mtx1, mtx2):
    D0, D1 = mtx1.shape
    D11, D2 = mtx2.shape
    if D11!=D1:
        raise ValueError("Shapes not aligned.")
    mtx = zeros((D0,D2))
    for j in range(D2):
        for i in range(D0):
            for k in range(D1):
                mtx[i,j] += mtx1[i,k] * mtx2[k,j]
    return mtx

def transpose(mtx1):
    D0, D1 = mtx1.shape
    mtx = zeros((D1,D0))
    for i in range(D0):
        for j in range(D1):
            mtx[j,i] = mtx1[i,j]
    return mtx

def argmax(mtx, axis = 0):
    D0, D1 = mtx.shape
    args = []
    if axis==0:
        for i in range(D0):
            arg = 0
            maxium = mtx[0,0]
            for j in range(D1):
                if mtx[i,j] > maxium:
                    maxium = mtx[i,j]
                    arg = j
            args.append(arg)
    else:
        for i in range(D1):
            arg = 0
            maxium = mtx[0,0]
            for j in range(D0):
                if mtx[j,i] > maxium:
                    maxium = mtx[j,i]
                    arg = j
            args.append(arg)
    return args

def sigmoid(mtx):
    D0, D1 = mtx.shape
    mtx1 = zeros(mtx.shape)
    for i in range(D0):
        for j in range(D1):
            mtx1[i,j] = 1 / (1 + math.exp(-mtx[i,j]))
    return mtx1

def log(mtx):
    D0, D1 = mtx.shape
    mtx1 = zeros(mtx.shape)
    for i in range(D0):
        for j in range(D1):
            mtx1[i,j] = math.log(mtx[i,j])
    return mtx1

def mean(mtx):
    D0, D1 = mtx.shape
    mean = 0
    for i in range(D0):
        for j in range(D1):
            mean += mtx[i,j]
    mean /= D0*D1
    return mean