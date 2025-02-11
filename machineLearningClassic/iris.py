import numpy as np
iris = np.genfromtxt("Iris.csv",delimiter=',',encoding=None)
iris = np.delete(iris,0,0)
iris_1 = np.delete(iris,(0,-1),1)
fin = np.delete(iris,slice(0,-1),1)
print(fin)
print(iris_1)

#Splitting the data into variables
x_sepal_length = iris.data[:,0]
x_sepal_width = iris.data[:,1]
x_petal_length = iris.data[:,2]
x_petal_width = iris.data[:,3]
y = iris.target

import numpy as np
X = iris["data"]
y = iris["target"]
percentage = 0.3
# y format is a vector with 3 different values "0, 1, 2" instead of the "versicolor, virginica, setosa" labels

def split(X,y,percentage):
    total = np.array(zip(X,y))
    print("total")
    print(total)
    print("total len")
    print(total.size)
    number_of_elements = int(total.size * percentage)
    test_indexes = np.random.choice(total.size, size=number_of_elements, replace=False)
    training_indexes = np.setdiff1d(np.arange(total.size), test_indexes)
    X_test,y_test = X[test_indexes],y[test_indexes]
    X_training,y_training = X[training_indexes],y[training_indexes]
    
    print("X_test")
    print(X_test)
    print("y_test")
    print(y_test)
    print("X_training")
    print(X_training)
    print("y_training")
    print(y_training)
    return X_test,X_training,y_test,y_training

