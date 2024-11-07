# mlp/data_loader.py

import pandas as pd
import numpy as np

def load_data(filename):
    """
    Carrega os dados de um arquivo CSV.

    Parâmetros:
    - filename (str): Caminho para o arquivo CSV.

    Retorna:
    - data (DataFrame): DataFrame contendo os dados carregados.
    """
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"O arquivo '{filename}' não foi encontrado.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"O arquivo '{filename}' está vazio.")
    except pd.errors.ParserError:
        raise ValueError(f"O arquivo '{filename}' não está no formato CSV correto.")
    except Exception as e:
        raise Exception(f"Erro ao carregar os dados: {e}")

def sample_data(data, percentage):
    """
    Seleciona uma porcentagem dos dados de um DataFrame.

    Parâmetros:
    - data (DataFrame): DataFrame de entrada.
    - percentage (float): Porcentagem dos dados a serem selecionados (0 < percentage <= 100).

    Retorna:
    - sampled_data (DataFrame): DataFrame com a amostra dos dados.
    """
    if not (0 < percentage <= 100):
        raise ValueError("A porcentagem deve estar entre 0 e 100.")
    sampled_data = data.sample(frac=percentage / 100.0, random_state=42)
    return sampled_data.reset_index(drop=True)
