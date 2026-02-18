import pandas as pd
import numpy as np
import torch
# import flautim as fl

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import FLLMDataset

from clshq_tk.modules.fuzzy import GridPartitioner, trimf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances

from transformers import DistilBertTokenizerFast


def fuzzification(df, name_dataset, letter, partitions, partitioner):

    ts = RegressionTS(name_dataset, 2000, df.values, order = 1, step_ahead = 1, dtype=torch.float64)

    if partitioner == None:
        partitioner = GridPartitioner(trimf, partitions, ts.num_attributes, device = DEVICE, dtype = ts.dtype,
                                    var_names = [letter])

    training_loop(partitioner, ts)

    fuzzy_X = partitioner.forward(ts.X.to(device=DEVICE), mode = 'one-hot')
    linguistic_X = np.array(partitioner.from_membership_to_linguistic(fuzzy_X))

    fuzzy_y = partitioner.forward(ts.y.to(device=DEVICE), mode = 'one-hot')
    linguistic_y = np.array(partitioner.from_membership_to_linguistic(fuzzy_y))

    return pd.DataFrame(linguistic_X), pd.DataFrame(linguistic_y), partitioner


import re
from collections import defaultdict

def completar_conjuntos_fuzzy(lista):
    grupos = defaultdict(set)

    # separa prefixo e índice numérico
    for item in lista:
        match = re.match(r"([A-Z]+)(\d+)", item)
        if match:
            prefixo, idx = match.group(1), int(match.group(2))
            grupos[prefixo].add(idx)

    resultado = []

    for prefixo, indices in grupos.items():
        min_i, max_i = min(indices), max(indices)
        for i in range(min_i, max_i + 1):
            resultado.append(f"{prefixo}{i}")

    return resultado

def fuzzy(df, lookback, partitions, partitioner, step_ahead, tokenizer, vocab, novos_tokens, max_length):

    variables = df.columns.tolist()

    # Fuzzification of time series
    X_fuzzy = dict.fromkeys(variables)  # para a entrada fuzzy de cada variável
    y_fuzzy = dict.fromkeys(variables)   # para a saída fuzzy de cada variável
    y_real = dict.fromkeys(variables)  # para armazenar a saída real de cada variável
    fuzzysets = []   # para armazenar todos os conjuntos fuzzy usados e adicionar ao tokenizer
    partitioner = dict.fromkeys(variables)
    for v in variables:
        X_fuzzy[v], y_fuzzy[v], partitioner[v] = fuzzification(pd.DataFrame(df[v]), "name_dataset", v, partitions, partitioner[v])
        y_real[v] = pd.DataFrame(df[v].iloc[:-2]).values.tolist()
        fuzzysets.append(set(X_fuzzy[v].values.flatten()))

    if tokenizer == None:
        # adiciona os conjuntos fuzzy como tokens ao tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        novos_tokens = [x for s in fuzzysets for x in s]
        print("Número de novos tokens antes da complementação: ", len(novos_tokens))
        novos_tokens = completar_conjuntos_fuzzy(novos_tokens)
        print("Número de novos tokens após a complementação: ", len(novos_tokens))
        tokenizer.add_tokens(novos_tokens)
        vocab = tokenizer.get_vocab()
    else:
        novos_tokens = []
        tokens_extra = list(set([x for s in fuzzysets for x in s]) - set(novos_tokens))
        if len(tokens_extra) > 0:
            tokenizer.add_tokens(tokens_extra)
            vocab = tokenizer.get_vocab()
            novos_tokens.extend(tokens_extra)

    X_variables = dict.fromkeys(variables)
    y_variables = dict.fromkeys(variables)
    labels = []
    input = []

    n = X_fuzzy[variables[0]].shape[0]-(lookback+step_ahead)
    for row in range(n):

        X_text = []
        y_text = []
        y = []
        for v in variables:
            X_variables[v] = X_fuzzy[v].iloc[row:row+lookback]
            X_text = str(X_variables[v].values.flatten().tolist()).replace("[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "")
            labels.append(np.array(y_real[v][lookback+row:lookback+row+step_ahead]).flatten().tolist())
            input.append(f"{X_text}")

    input_tokens = tokenizer(
        input,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True
    )


    return FLLMDataset.FLLMDataset(input_tokens['input_ids'], input_tokens['attention_mask'], labels), tokenizer, partitioner, vocab, novos_tokens