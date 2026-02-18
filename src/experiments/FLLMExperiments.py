import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import pandas as pd
import common

def mean_absolute_error(y, y_hat):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    return np.mean(np.abs(y - y_hat))

def mean_squared_error(y, y_hat):
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    return np.mean((y - y_hat) ** 2)

class FLLMExperiments():
    def __init__(self, context, model, **kwargs):
        # super(FLLMExperiments, self).__init__(model, kwargs["dataset_train"], context, **kwargs)
        self.dataset_train = kwargs["dataset_train"]
        self.dataset_val = kwargs["dataset_val"]
        self.dataset_test = kwargs["dataset_test"]
        self.pca = kwargs['pca']
        self.gamma = kwargs['gamma']
        self.variables = kwargs['variables']
        self.embedding = kwargs['embedding']
        self.model = model
        self.epochs = kwargs['epochs']
        self.use_lora = kwargs['use_lora']
        self.step_ahead = kwargs['step_ahead']
        self.output_size = kwargs['output_size']
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def fit(self, **kwargs):

        print(f"Model training started:")
        self.train_model()

        print(f"Predict")

        if self.embedding:
            forecasts, real = self.predict()
            forecasts_inverse = []
            for i in range(0, forecasts.shape[1], int(forecasts.shape[1]/self.step_ahead)):
                forecasts_inverse.append(common.inverse_kpca_transformation(forecasts.iloc[:, i:i+self.pca.no_of_components], self.pca, self.gamma))

            forecasts_inverse = [pd.DataFrame(arr) for arr in forecasts_inverse]
            forecasts = pd.concat(forecasts_inverse, axis=1)

            forecasts.columns = self.lst_sort
        else:
            forecasts, real = self.predict()
            # print(forecasts)
            # print(real)

        print("Métricas calculadas a partir da série média")
        forecasts_mean = forecasts.mean(axis=1).to_frame(name='mean_value')
        real_mean = real.mean(axis=1).to_frame(name='mean_value')
        print(f"MAE: {mean_absolute_error(real_mean, forecasts_mean)}")
        print(f"MSE: {mean_squared_error(real_mean, forecasts_mean)}")

        return forecasts, real

    def train_model(self):

        # Model to device
        self.model.to(self.DEVICE)

        training_args = TrainingArguments(
            output_dir="./models",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            learning_rate = 1e-5 if self.use_lora else 5e-5,
            weight_decay=0.001,   # avoid hurting embeddings
            logging_steps=20,
            report_to="none",
            save_strategy="no",
            save_total_limit=1,
            save_safetensors=False,
            fp16=torch.cuda.is_available()
        )

        optimizer = torch.optim.AdamW([
            {"params": self.model.model.embeddings.word_embeddings.parameters(), "lr": 1e-3},
            {"params": self.model.mlp_head.parameters(), "lr": 3e-4}
        ])

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_val
        )

        trainer.train()


    def predict(self):

        dataloader = DataLoader(self.dataset_test, batch_size=32, shuffle=False)

        all_preds = pd.DataFrame()
        all_actuals = pd.DataFrame()

        self.model.eval()

        with torch.no_grad():

            for batch_data in dataloader:
                inputs = batch_data['input_ids'].to(self.DEVICE)
                attention_mask = batch_data['attention_mask'].to(self.DEVICE)
                labels = batch_data['labels'].to(self.DEVICE)

                outputs = self.model.forward(inputs, attention_mask=attention_mask, labels=None)

                predictions = outputs['logits'].detach().cpu()
                true = labels.detach().cpu()

                unscaled, trues = [], []
                for i in range(self.output_size):
                    unscaled.append(predictions[:,i].reshape(-1, 1))
                    trues.append(true[:,i].reshape(-1, 1))


                batch_preds_df = pd.DataFrame(np.column_stack(unscaled))
                batch_trues_df = pd.DataFrame(np.column_stack(trues))


                all_preds = pd.concat([all_preds, batch_preds_df], ignore_index=True)
                all_actuals = pd.concat([all_actuals, batch_trues_df], ignore_index=True)

        return all_preds, all_actuals