# mlp/gui.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import threading
from .model import MLP
from .data_loader import load_data, sample_data
from sklearn.metrics import confusion_matrix
import numpy as np
import logging
import queue

class MLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rede Neural MLP com Backpropagation")
        
        # Configurar o logging
        logging.basicConfig(filename='mlp_app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Aplicação iniciada.")
        
        # Variáveis da Interface
        self.hidden_neurons_var = tk.StringVar(value='5')
        self.error_threshold_var = tk.StringVar(value='0.01')
        self.epochs_var = tk.StringVar(value='1000')
        self.learning_rate_var = tk.StringVar(value='0.1')
        self.activation_function_var = tk.StringVar(value='linear')
        self.status_var = tk.StringVar(value='Pronto')

        # Variáveis de dados
        self.consecutive_no_change_epochs = 0
        self.previous_mse = None
        self.max_no_change_epochs = 10


        # Inicialização das variáveis de dados
        self.training_filepath = 'data/treinamento.csv'
        self.testing_filepath = 'data/teste.csv'
        self.single_file_path = 'data/arquivo_unico.csv'
        
        # Variáveis para a escolha da fonte dos dados
        self.data_source_var = tk.StringVar(value='two_files')
        
        # Variáveis para a janela do gráfico
        self.graph_window = None
        self.figure = None
        self.ax = None
        self.canvas = None

        # Criar uma fila para comunicação entre threads
        self.queue = queue.Queue()
        # Iniciar o processamento da fila
        self.root.after(100, self.process_queue)

        # Montar a Interface
        self.create_widgets()
        
    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)
        
        # Opções de seleção de fonte de dados
        data_source_frame = tk.LabelFrame(frame, text="Fonte dos Dados")
        data_source_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky='we')
        
        tk.Radiobutton(data_source_frame, text="Usar dois arquivos (treinamento e teste)", variable=self.data_source_var, value='two_files', command=self.update_data_source).grid(row=0, column=0, sticky='w')
        tk.Radiobutton(data_source_frame, text="Usar um arquivo único", variable=self.data_source_var, value='single_file', command=self.update_data_source).grid(row=1, column=0, sticky='w')
        
        # Campo para o número de neurônios na camada oculta
        tk.Label(frame, text="Neurônios na Camada Oculta:").grid(row=2, column=0, sticky='e')
        tk.Entry(frame, textvariable=self.hidden_neurons_var).grid(row=2, column=1)
        
        # Campo para o valor de erro (limiar)
        tk.Label(frame, text="Valor de Erro (Opcional):").grid(row=3, column=0, sticky='e')
        tk.Entry(frame, textvariable=self.error_threshold_var).grid(row=3, column=1)
        
        # Campo para o número de iterações (épocas)
        tk.Label(frame, text="Número de Iterações:").grid(row=4, column=0, sticky='e')
        tk.Entry(frame, textvariable=self.epochs_var).grid(row=4, column=1)
        
        # Campo para a taxa de aprendizado (N)
        tk.Label(frame, text="Taxa de Aprendizado (N):").grid(row=5, column=0, sticky='e')
        tk.Entry(frame, textvariable=self.learning_rate_var).grid(row=5, column=1)
        
        # Campo para a função de transferência
        tk.Label(frame, text="Função de Transferência:").grid(row=6, column=0, sticky='e')
        activation_options = ['linear', 'logistic', 'tanh']
        tk.OptionMenu(frame, self.activation_function_var, *activation_options).grid(row=6, column=1, sticky='we')
        
        # Frame para as porcentagens dos dados
        self.data_percentage_frame = tk.LabelFrame(frame, text="Porcentagem dos Dados (Amostragem)")
        self.data_percentage_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky='we')
        
        tk.Label(self.data_percentage_frame, text="Porcentagem de Treinamento (%):").grid(row=0, column=0, sticky='e')
        self.train_percentage_var = tk.StringVar(value='70')
        tk.Entry(self.data_percentage_frame, textvariable=self.train_percentage_var, width=5).grid(row=0, column=1, sticky='w')
        
        tk.Label(self.data_percentage_frame, text="Porcentagem de Teste (%):").grid(row=1, column=0, sticky='e')
        self.test_percentage_var = tk.StringVar(value='30')
        tk.Entry(self.data_percentage_frame, textvariable=self.test_percentage_var, width=5).grid(row=1, column=1, sticky='w')
        
        # Desabilitar os widgets dentro do frame de porcentagens inicialmente se a opção selecionada for usar um arquivo único
        if self.data_source_var.get() == 'single_file':
            self.set_data_percentage_frame_state('disabled')
        else:
            self.set_data_percentage_frame_state('normal')
        
        # Botão para iniciar o treinamento
        tk.Button(frame, text="Iniciar Treinamento", command=self.start_training).grid(row=8, column=0, columnspan=2, pady=10)
        
        # Label de status
        tk.Label(frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=2)
        
        # Área para exibir os dados de treinamento
        tk.Label(self.root, text="Dados de Treinamento (Normalizados):").pack(pady=(10, 0))
        self.train_tree = ttk.Treeview(self.root, show='headings')
        self.train_tree.pack(fill='both', expand=True, padx=10)
        
        # Área para exibir os dados de teste
        tk.Label(self.root, text="Dados de Teste (Normalizados):").pack(pady=(10, 0))
        self.test_tree = ttk.Treeview(self.root, show='headings')
        self.test_tree.pack(fill='both', expand=True, padx=10)
        
    def set_data_percentage_frame_state(self, state):
        for child in self.data_percentage_frame.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass  # Widget não suporta a opção 'state'

    def update_data_source(self):
        # Habilitar ou desabilitar os campos de porcentagem com base na escolha do usuário
        if self.data_source_var.get() == 'single_file':
            self.set_data_percentage_frame_state('disabled')
        else:
            self.set_data_percentage_frame_state('normal')
        
    def display_data(self, data, tree):
        # Limpar a Treeview
        tree.delete(*tree.get_children())
        tree['columns'] = list(data.columns)
        # Definir os cabeçalhos
        for col in data.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        # Inserir os dados
        for index, row in data.iterrows():
            tree.insert("", "end", values=list(row))
    
    def start_training(self):
        # Fechar a janela do gráfico anterior, se existir
        if self.graph_window is not None:
            self.on_close_graph_window()
        
        # Reinicializar a fila
        self.queue = queue.Queue()
        # Iniciar o processamento da fila novamente
        self.root.after(100, self.process_queue)
        
        try:
            # Obter os parâmetros da interface
            hidden_neurons = int(self.hidden_neurons_var.get())
            if hidden_neurons <= 0:
                raise ValueError("O número de neurônios deve ser positivo.")
            
            error_threshold_input = self.error_threshold_var.get()
            if error_threshold_input:
                error_threshold = float(error_threshold_input)
                if error_threshold <= 0:
                    raise ValueError("O valor de erro deve ser positivo.")
            else:
                error_threshold = None
            
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError("O número de iterações deve ser positivo.")
            
            learning_rate = float(self.learning_rate_var.get())
            if not (0 < learning_rate <= 1):
                raise ValueError("A taxa de aprendizado deve estar entre 0 e 1.")
            
            activation_function = self.activation_function_var.get()
            if activation_function not in ['linear', 'logistic', 'tanh']:
                raise ValueError("Função de ativação inválida.")
            
            # Atualizar o status
            self.status_var.set("Carregando os dados...")
            
            # Carregar os dados com base na escolha do usuário
            if self.data_source_var.get() == 'single_file':
                # Usar todo o conjunto de dados do arquivo único
                data_full = load_data(self.single_file_path)
                training_data = data_full
                testing_data = data_full  # Opcionalmente, você pode definir testing_data como vazio ou usar o mesmo conjunto
            else:
                # Obter as porcentagens dos dados
                train_percentage = float(self.train_percentage_var.get())
                test_percentage = float(self.test_percentage_var.get())
                if not (0 < train_percentage <= 100):
                    raise ValueError("A porcentagem de treinamento deve estar entre 0 e 100.")
                if not (0 < test_percentage <= 100):
                    raise ValueError("A porcentagem de teste deve estar entre 0 e 100.")
                if (train_percentage + test_percentage) > 100:
                    raise ValueError("A soma das porcentagens de treinamento e teste não pode exceder 100%.")
                
                # Carregar os dados
                training_data_full = load_data(self.training_filepath)
                testing_data_full = load_data(self.testing_filepath)
                # Selecionar a porcentagem dos dados
                training_data = sample_data(training_data_full, train_percentage)
                testing_data = sample_data(testing_data_full, test_percentage)
            
            # Combinar os dados para codificação
            all_data = pd.concat([training_data, testing_data], ignore_index=True)
            
            # Codificar as classes
            X_all = all_data.iloc[:, :-1].values.astype(float)
            y_raw_all = all_data.iloc[:, -1].values
            classes = np.unique(y_raw_all)
            class_to_num = {k: v for v, k in enumerate(classes)}
            y_num_all = np.array([class_to_num[cls] for cls in y_raw_all])
            Y_all = np.eye(len(classes))[y_num_all]
            
            # Separar novamente os dados de treinamento e teste
            X_train = X_all[:len(training_data)]
            Y_train = Y_all[:len(training_data)]
            y_train_num = y_num_all[:len(training_data)]
            
            X_test = X_all[len(training_data):]
            Y_test = Y_all[len(training_data):]
            y_test_num = y_num_all[len(training_data):]
            
            # Normalização dos dados com base no conjunto de treinamento
            min_values = X_train.min(axis=0)
            max_values = X_train.max(axis=0)
            ranges = max_values - min_values
            ranges[ranges == 0] = 1  # Evitar divisão por zero
            X_train_normalized = (X_train - min_values) / ranges
            X_test_normalized = (X_test - min_values) / ranges
            
            # Criar DataFrames para exibir na interface
            y_train_raw = [classes[num] for num in y_train_num]
            y_test_raw = [classes[num] for num in y_test_num]
            
            data_train_normalized = pd.DataFrame(X_train_normalized, columns=training_data.columns[:-1])
            data_train_normalized['classe'] = y_train_raw
            data_test_normalized = pd.DataFrame(X_test_normalized, columns=testing_data.columns[:-1])
            data_test_normalized['classe'] = y_test_raw
            
            # Atualizar variáveis de instância
            self.X_train = X_train_normalized
            self.Y_train = Y_train
            self.y_train_num = y_train_num
            self.class_to_num = class_to_num
            self.min_values = min_values
            self.max_values = max_values
            self.data_train_normalized = data_train_normalized
            
            self.X_test = X_test_normalized
            self.Y_test = Y_test
            self.y_test_num = y_test_num
            self.data_test_normalized = data_test_normalized
            
            # Atualizar as tabelas na interface
            self.display_data(self.data_train_normalized, self.train_tree)
            self.display_data(self.data_test_normalized, self.test_tree)
            
            # Atualizar o tamanho das camadas de entrada e saída
            self.input_size = self.X_train.shape[1]
            self.output_size = self.Y_train.shape[1]
            
            # Atualizar o status
            self.status_var.set("Preparando o gráfico...")
            
            # Criar a janela do gráfico
            self.create_graph_window()
            
            # Atualizar o status
            self.status_var.set("Treinando a rede...")
            
            # Inicializar e treinar a rede em uma thread separada
            self.mlp = MLP(self.input_size, hidden_neurons, self.output_size, learning_rate, activation_function)
            training_thread = threading.Thread(target=self.train_network, args=(epochs, error_threshold))
            training_thread.start()
            logging.info(f"Iniciando treinamento: Neurônios Ocultos={hidden_neurons}, Épocas={epochs}, Taxa de Aprendizado={learning_rate}, Função de Ativação={activation_function}")
        except ValueError as ve:
            messagebox.showerror("Erro de Validação", str(ve))
            logging.error(f"Erro de Validação: {ve}")
            self.status_var.set("Erro ao iniciar o treinamento.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar o treinamento: {e}")
            logging.error(f"Erro ao iniciar o treinamento: {e}")
            self.status_var.set("Erro ao iniciar o treinamento.")
    
    def create_graph_window(self):
        # Criar uma nova janela para o gráfico
        self.graph_window = tk.Toplevel(self.root)
        self.graph_window.title("Progressão do Erro")
        self.graph_window.protocol("WM_DELETE_WINDOW", self.on_close_graph_window)
        
        # Configurar o gráfico
        self.figure = plt.Figure(figsize=(6,4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Progressão do Erro Durante o Treinamento')
        self.ax.set_xlabel('Épocas')
        self.ax.set_ylabel('Erro Quadrático Médio (MSE)')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def on_close_graph_window(self):
        # Função para fechar a janela do gráfico
        if self.graph_window:
            self.graph_window.destroy()
            self.graph_window = None

    def train_network(self, epochs, error_threshold):
        def update_progress(epoch, mse):
            if self.previous_mse is not None:
                print(self.previous_mse - mse)
                if abs(self.previous_mse - mse) < 0.001:  # Definir um limite para considerar "sem mudança"
                    self.consecutive_no_change_epochs += 1
                else:
                    self.consecutive_no_change_epochs = 0
            self.previous_mse = mse

            # Verificar se atingiu o platô
            if self.consecutive_no_change_epochs >= self.max_no_change_epochs:
                self.queue.put('PLATEAU_REACHED')

            # Colocar o progresso na fila
            self.queue.put((epoch, mse))

        self.mlp.train(self.X_train, self.Y_train, max_epochs=epochs, error_threshold=error_threshold, update_callback=update_progress)
        
        # Após o treinamento, colocar um valor sentinela na fila
        self.queue.put('TRAINING_COMPLETED')

    def process_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                print('item', item)
                if item == 'TRAINING_COMPLETED':
                    self.after_training()
                    return
                elif item == 'PLATEAU_REACHED':
                    self.ask_user_to_adjust_learning_rate()
                    return
                else:
                    epoch, mse = item
                    self.update_graph(epoch, mse)
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)


    def update_graph(self, epoch, mse):
        # Atualizar o gráfico na janela separada
        if self.graph_window:
            self.ax.clear()
            # Garantir que os arrays tenham o mesmo comprimento
            error_history = self.mlp.error_history[:epoch]
            self.ax.plot(range(1, len(error_history) + 1), error_history, label='Erro Quadrático Médio', color='blue')
            self.ax.set_title('Progressão do Erro Durante o Treinamento')
            self.ax.set_xlabel('Épocas')
            self.ax.set_ylabel('Erro Quadrático Médio (MSE)')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()
            # Atualizar o status
            self.status_var.set(f'Época {epoch}, Erro: {mse:.6f}')
            logging.info(f'Época {epoch}, Erro: {mse:.6f}')

    def after_training(self):
        # Função chamada após o treinamento ser concluído
        # Realizar as previsões e mostrar a matriz de confusão
        predictions = self.mlp.predict(self.X_test)
        
        # Mapeamento dos números de volta para as classes originais
        num_to_class = {v: k for k, v in self.class_to_num.items()}
        predicted_classes = [num_to_class[num] for num in predictions]
        actual_classes = [num_to_class[num] for num in self.y_test_num]
        
        # Geração da matriz de confusão
        conf_matrix = confusion_matrix(actual_classes, predicted_classes, labels=list(self.class_to_num.keys()))
        
        # Exibir a matriz de confusão em uma nova janela
        self.show_confusion_matrix(conf_matrix, list(self.class_to_num.keys()))
        
        # Atualizar o status
        self.status_var.set("Treinamento concluído.")
        logging.info("Treinamento concluído.")

    def show_confusion_matrix(self, conf_matrix, class_names):
        # Criar uma nova janela para a matriz de confusão
        cm_window = tk.Toplevel(self.root)
        cm_window.title("Matriz de Confusão")
        
        # Configurar o gráfico da matriz de confusão
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Definir ticks e labels
        ax.set(
            xticks=np.arange(conf_matrix.shape[1]),
            yticks=np.arange(conf_matrix.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel='Classe Real',
            xlabel='Classe Predita',
            title='Matriz de Confusão'
        )
        
        # Rotacionar os labels do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar os valores da matriz de confusão nas células
        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Exibir o gráfico na nova janela
        canvas = FigureCanvasTkAgg(fig, master=cm_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Opcional: Adicionar um botão para fechar a janela
        tk.Button(cm_window, text="Fechar", command=cm_window.destroy).pack(pady=5)

    def ask_user_to_adjust_learning_rate(self):
        self.mlp.stop_training()
        response = messagebox.askyesno(
            "Platô Atingido",
            "O erro médio não mudou em 10 épocas consecutivas.\n"
            "Deseja ajustar a taxa de aprendizado e continuar o treinamento?"
        )
        if response:
            new_rate = simpledialog.askfloat(
                "Nova Taxa de Aprendizado",
                "Digite a nova taxa de aprendizado (0 < taxa ≤ 1):",
                minvalue=0.01, maxvalue=1.0
            )
            if new_rate:
                self.mlp.learning_rate = new_rate
                self.consecutive_no_change_epochs = 0
                self.learning_rate_var.set(str(new_rate))
                self.mlp.resume_training()
                self.root.after(100, self.process_queue)
        else:
            self.status_var.set("Treinamento interrompido pelo usuário.")
            logging.info("Treinamento interrompido pelo usuário devido ao platô.")
            self.mlp.finished_training = True
            self.queue.put('TRAINING_COMPLETED')
            self.root.after(100, self.process_queue)

