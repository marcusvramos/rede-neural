# 🧠 Rede Neural MLP com Backpropagation e Interface Gráfica

Este repositório contém uma implementação em Python de uma Rede Neural Perceptron Multicamadas (MLP) com algoritmo de backpropagation. A aplicação possui uma interface gráfica desenvolvida com Tkinter, permitindo:

- ⚙️ **Configuração de Parâmetros**: Ajuste do número de neurônios na camada oculta, taxa de aprendizado, número de épocas, função de ativação (linear, logística ou tangente hiperbólica) e definição do erro desejado.
- 📊 **Visualização dos Dados**: Exibição dos dados de treinamento e teste após normalização.
- 🚀 **Treinamento em Tempo Real**: Acompanhamento do progresso do treinamento com atualização em tempo real do gráfico de erro quadrático médio (MSE).
- 🧐 **Avaliação do Modelo**: Geração e visualização da matriz de confusão após o treinamento para análise de desempenho.

## ✨ Funcionalidades Principais

- 🖥️ **Interface Amigável**: Interface intuitiva para facilitar a interação com a rede neural.
- 📈 **Gráfico de Erro Dinâmico**: Visualização animada do erro ao longo das épocas, permitindo monitorar a convergência do modelo.
- 🔧 **Personalização**: Flexibilidade para ajustar os hiperparâmetros conforme a necessidade do usuário.
- 📊 **Matriz de Confusão**: Análise de desempenho através de uma matriz de confusão exibida em uma janela separada após o treinamento.

## 📂 Estrutura do Projeto

- `mlp/model.py`: Implementação da estrutura da Rede Neural MLP com backpropagation.
- `mlp/gui.py`: Interface gráfica que permite interação com a rede neural.
- `mlp/data_loader.py`: Funções de carregamento e amostragem dos dados para treinamento e teste.
- `data/`: Pasta com arquivos `treinamento.csv` e `teste.csv` contendo dados de exemplo.

## 🚀 Como Usar

1. Clone o repositório:
```bash
   git clone https://github.com/seu_usuario/rede-neural-mlp.git
```

2. Instale as dependências:
```bash
   pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
   python main.py
```

## 🛠️ Dependências

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tkinter` (disponível nativamente no Python)

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e fazer pull requests.

