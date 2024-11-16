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
        self.is_training = True
        self.finished_training = False

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

    def train(self, X, Y, max_epochs=1000, error_threshold=None, update_callback=None, complete_callback=None):
        self.error_history = []
        epoch = 0
        error_mean = float('inf')

        while not self.finished_training and epoch < max_epochs and error_mean > error_threshold if error_threshold else True:
            if self.is_training:
                error_mean = 0
                epoch += 1

                for i in range(X.shape[0]):
                    input_data = X[i:i+1]  # Seleciona o exemplo atual (uma única linha)
                    target = Y[i:i+1]

                    # Propagação para frente
                    output = self.forward(input_data)

                    # Calcula o erro da rede para este exemplo
                    example_error = np.sum((target - output) ** 2) / 2
                    error_mean += example_error

                    # Propagação para trás
                    self.backward(input_data, target)

                # Calcula o erro médio ao final da época
                error_mean /= X.shape[0]
                self.error_history.append(error_mean)

                # Callback de atualização
                if update_callback and error_threshold and (epoch % 10 == 0 or error_mean <= error_threshold):
                    update_callback(epoch, error_mean)

                if error_threshold and error_mean <= error_threshold:
                    print(f'Treinamento parado na época {epoch} com erro médio {error_mean}')
                    break

        # Finaliza o treinamento
        if update_callback:
            update_callback(epoch, error_mean)

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

    def stop_training(self):
        self.is_training = False

    def resume_training(self):
        self.is_training = True