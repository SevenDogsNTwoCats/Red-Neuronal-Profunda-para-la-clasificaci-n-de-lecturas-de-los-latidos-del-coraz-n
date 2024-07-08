import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import OneHotEncoder

#cargar parametros
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

#cargar parametros
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')


#One Hot encoding
enc = OneHotEncoder(sparse=False, categories='auto')
train_y = enc.fit_transform(y_train.reshape(len(y_train), -1))
test_y = enc.transform(y_train.reshape(len(y_train), -1))
# arreglando las dimensiones
train_y = train_y.reshape(train_y.shape[1], train_y.shape[0])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[0])
print(x_train.shape)
print(train_y.shape)

#funciones de activacion
def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A

def softmax(z):
    e = np.exp(z - np.max(z))
    s = e / e.sum(axis=0, keepdims=True)
    assert(z.shape == s.shape)
    return s

#inicializar w y b
def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    for l in range(1, L):
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        #probando la inicializacion de He que supuestamente es mas efectiva para ReLU ya que el costo no bajaba mucho
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])

        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

#propagacion hacia adelante
def linear_forward(A, W, b):
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        Z = linear_forward(A_prev, W, b)
        A = softmax(Z)
    elif activation == "relu":
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (Z, A_prev, W)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    A_prev = A
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, "softmax")
    caches.append(cache)
    assert(AL.shape == (W.shape[0], X.shape[1]))
    return AL, caches

#calculo del costo
def compute_cost(AL, Y):
    cost = - np.mean(Y*np.log(AL + 1e-8))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

#propagacion hacia atras
def linear_backward(dZ, cache):
    Z, A_prev, W = cache # Z is not needed
    m = A_prev.shape[1]
    dW = 1/m * dZ.dot(A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (W.T).dot(dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape) 
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    Z, A_prev, W = cache

    if activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dA_prev, dW, db = linear_backward(dZ, cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    # Initializing the backpropagation
    # dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))
    dAL =-(Y/AL)+(1-Y)/(1-AL)
    dZ = AL -Y
    # Lth layer (SOFTMAX -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache=caches[-1]
    Z, A_prev_temp, W = current_cache
    dW = dZ.dot(A_prev_temp.T) /m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T.dot(dZ)

    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
             linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

#descenso de gradiente
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network
    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

#normalizar X
def normalizeRows(x):
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims = True)
    x = x/x_norm

    return x

#aplicacion de todas las funciones en un modelo
def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)

    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        
        # Compute cost.
        cost = - np.mean(Y*np.log(AL + 1e-8))

    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)       
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    #guardar en el json
    params = {
        "dnn_layers": []
    }
    for l in range(1, len(layers_dims)):
        layer_params = {
            "n": layers_dims[l],
            "activation": "relu" if l < len(layers_dims) - 1 else "softmax",
            "w": parameters["W" + str(l)].tolist(),
            "b": parameters["b" + str(l)].flatten().tolist()
        }
        params["dnn_layers"].append(layer_params)
        combined_b = np.concatenate([parameters["b" + str(l)].flatten() for l in range(1, len(layers_dims))]).tolist()
        params["combined_b"] = combined_b

        with open('params.json', 'w') as json_file:
            json.dump(params, json_file, indent=4)
    
    return parameters

# predecir
def predict(X, Y, parameters):
        A, cache = L_model_forward(X, parameters)
        y_hat = np.argmax(A.T, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100





#pruebas
layers_dims = [x_train.shape[0], 15,15, train_y.shape[0]]
train_x = normalizeRows(x_train)
test_x = normalizeRows(x_test)
parameters = L_layer_model(train_x, train_y, layers_dims, 0.09, 5000,True)