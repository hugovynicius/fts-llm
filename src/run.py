import pandas as pd
import numpy as np
import common
# import flautim as fl
from peft import LoraConfig, TaskType
import FLLMModel, FLLMExperiments
import utils

if __name__ == '__main__':

    # context = fl.init()

    # print(f"Flautim inicializado!!!")

    # Upload do dataset
    path = 'ETTh1.csv'
    root_path = ''
    name_dataset = 'ETTh1'
    train_data, train_loader = utils.get_data(name_dataset=name_dataset, root_path=root_path, path=path, flag='train')
    val_data, val_loader = utils.get_data(name_dataset=name_dataset, root_path=root_path, path=path, flag='val')
    test_data, test_loader = utils.get_data(name_dataset=name_dataset, root_path=root_path, path=path, flag='test')

    name_variables = ['A','B','C','D','E','F','G']
    df_train = pd.DataFrame(train_data.data_x, columns=name_variables)
    df_val = pd.DataFrame(val_data.data_x, columns=name_variables)
    df_test = pd.DataFrame(test_data.data_x, columns=name_variables)

    lookback = 96
    partitions = 60
    epochs = 5
    step_ahead = 96
    embedding = False
    output_size = step_ahead
    context = None
    partitioner = None
    tokenizer = None
    vocab = None
    max_length = 100
    novos_tokens = []
    ds_train, tokenizer, partitioner, vocab, novos_tokens = common.fuzzy(df_train, lookback, partitions, partitioner, step_ahead, tokenizer, vocab, novos_tokens, max_length)
    ds_val, tokenizer, partitioner, vocab, novos_tokens = common.fuzzy(df_val, lookback, partitions, partitioner, step_ahead, tokenizer, vocab, novos_tokens, max_length)
    ds_test, tokenizer, partitioner, vocab, novos_tokens = common.fuzzy(df_test, lookback, partitions, partitioner, step_ahead, tokenizer, vocab, novos_tokens, max_length)

    model = FLLMModel.FLLMModel(context, output_size, tokenizer, hidden_dims=[128,64])
    
    experiment = FLLMExperiments.FLLMExperiments(context, model,
                                    dataset_train = ds_train,
                                    dataset_val = ds_val,
                                    dataset_test = ds_test,
                                    pca=None,
                                    gamma=None,
                                    variables=list(df_train.columns),
                                    embedding=embedding,
                                    name_model="distilbert-base-uncased",
                                    epochs=epochs,
                                    use_lora=True,
                                    step_ahead=step_ahead,
                                    output_size=output_size
    )

    forecasts, real = experiment.fit()