# mlp/model.py

import numpy as np
import pickle

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation_function):
        # Inicialização dos parâmetros da rede
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        
        # Pesos entre a camada de entrada e a camada escondida
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        # Pesos entre a camada escondida e a camada de saída
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        
        # Biases
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        # Histórico do erro
        self.error_history = []

    def activation(self, x):
        # Funções de ativação
        if self.activation_function == 'linear':
            return x / 10
        elif self.activation_function == 'logistic':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)

    def activation_derivative(self, x):
        # Derivadas das funções de ativação
        if self.activation_function == 'linear':
            return 1 / 10
        elif self.activation_function == 'logistic':
            fx = self.activation(x)
            return fx * (1 - fx)
        elif self.activation_function == 'tanh':
            fx = self.activation(x)
            return 1 - fx ** 2

    def forward(self, X):
        # Propagação para a camada escondida
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        # Propagação para a camada de saída
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.activation(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = Y.shape[0]
        # Erro na camada de saída
        error_output = (Y - self.A2) * self.activation_derivative(self.Z2)
        # Erro na camada escondida
        error_hidden = np.dot(error_output, self.W2.T) * self.activation_derivative(self.Z1)
        
        # Atualização dos pesos e biases
        self.W2 += self.learning_rate * np.dot(self.A1.T, error_output) / m
        self.b2 += self.learning_rate * np.sum(error_output, axis=0, keepdims=True) / m
        self.W1 += self.learning_rate * np.dot(X.T, error_hidden) / m
        self.b1 += self.learning_rate * np.sum(error_hidden, axis=0, keepdims=True) / m

    def train(self, X, Y, epochs=1000, error_threshold=None, update_callback=None):
        self.error_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y)
            # Cálculo do erro quadrático médio
            mse = np.mean((Y - output) ** 2)
            self.error_history.append(mse)
            if error_threshold and mse < error_threshold:
                print(f'Treinamento parado na época {epoch+1} com erro {mse}')
                if update_callback:
                    update_callback(epoch+1, mse)
                break
            if update_callback and (epoch+1) % 10 == 0:
                update_callback(epoch+1, mse)
        else:
            # Se o loop não foi interrompido pelo break
            if update_callback:
                update_callback(epochs, mse)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'W1': self.W1,
                'W2': self.W2,
                'b1': self.b1,
                'b2': self.b2,
                'error_history': self.error_history
            }, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.W1 = data['W1']
            self.W2 = data['W2']
            self.b1 = data['b1']
            self.b2 = data['b2']
            self.error_history = data['error_history']
