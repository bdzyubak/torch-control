from typing import Union, Optional

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as pl
import mlflow

from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification, BertForSequenceClassification,
                          BertTokenizer, RobertaTokenizer, RobertaModel, AutoModelForCausalLM, AutoTokenizer)

from services.training_setup import trainer_setup
from torch_utils import clear_layers, freeze_layers, get_model_param_num
from utils.torch_utils import tensor_to_numpy, average_round_metric

# NB: Speed up processing for negligible loss of accuracy. Verify acceptable accuracy for a production use case
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.allow_tf32 = True

supported_models = {'distilbert': 'distilbert-base-uncased', 'bert': 'bert-base-uncased',
                    'roberta': 'FacebookAI/roberta-base', 'llama': 'TheBloke/llama-2-70b-Guanaco-QLoRA-fp16'}


class FineTuneLLMAsClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, device: str = 'cuda:0', learning_rate: float = 5e-5,
                 do_layer_freeze: bool = True, extra_class_layers: Optional[Union[int, list]] = None,
                 fine_tune_dropout_rate: float = 0):
        """
        A Pytorch Lightning wrapper for supported LLM models for classification to simplify training and integrate with
        MLFlow
        Args:
            model_name: Name of supported model. Check supported_models on top of this module
            num_classes: The number of output classes to classify. Determines classifier head connections
            device: 'cpu' or 'cuda:[number]'
            learning_rate: The optimizer's starting learning rate
            do_layer_freeze: Freeze all layers except default and added classifier layers. Gets very high train accuracy
            but poor, decreasing val accuracy if off. The checkpointer will still save the best val epoch, so this is
            definitely an option. do_layer_freeze=False trains much faster due to small number of epochs, and need to
            propagate through all layers, even when do_layer_freeze=True
            extra_class_layers: List of connections for fully connected layers to add, or integer number of layers
            to add with default number of connections
        """

        super(FineTuneLLMAsClassifier, self).__init__()

        self.extra_class_layers = extra_class_layers
        self.fine_tune_dropout_rate = fine_tune_dropout_rate

        self.set_up_model_and_tokenizer(device, do_layer_freeze, model_name, num_classes)

        self.learning_rate = learning_rate
        mlflow.log_params({'model_name': model_name,
                           'num_classes': num_classes,
                           'learning_rate': learning_rate,
                           'do_layer_freeze': do_layer_freeze,
                           'extra_class_layers': extra_class_layers,
                           'fine_tune_dropout_rate': fine_tune_dropout_rate})

    def set_up_model_and_tokenizer(self, device, do_layer_freeze, model_name, num_classes):
        check_model_supported(model_name)
        # TODO: explore swapping tokenizers. For now, use native
        # WARNING: The fine_tune_head layers must be in order, as the input_features are used to replace them when
        # extra_class_layers is used
        if model_name == 'distilbert':
            model = DistilBertForSequenceClassification.from_pretrained(supported_models['distilbert'],
                                                                        num_labels=num_classes)
            self.fine_tune_head = ['pre_classifier', 'classifier']
            # Hardcode for now
            self.fine_tune_input_layers = 768

        elif model_name == 'bert':
            model = BertForSequenceClassification.from_pretrained(supported_models['bert'], num_labels=num_classes)
            self.fine_tune_head = ['classifier.bias', 'classifier.weight']
        elif model_name == 'roberta':
            raise NotImplementedError(f"Trainer needs to pass arguments differently. labels keyword not found.")
            model = RobertaModel.from_pretrained(supported_models['roberta'],
                                                 num_labels=num_classes)
            self.fine_tune_head = ['pooler.dense.bias', 'pooler.dense.weight']
        elif model_name.lower() == 'llama':
            raise NotImplementedError(f"Not tested.")
            model = AutoModelForCausalLM.from_pretrained(supported_models['llama'],
                                                         num_classes=num_classes)
            last_layer = ''  # Add this
        else:
            raise NotImplementedError(f"Support for the model {model_name} has not been implemented.")

        self.model = model
        if self.extra_class_layers:
            if isinstance(self.extra_class_layers, int):
                # Fill with default number of connections. Otherwise, expect list of connections
                self.extra_class_layers = [self.fine_tune_input_layers] * self.extra_class_layers
            else:
                for ind, layer in enumerate(self.extra_class_layers):
                    if layer < self.fine_tune_input_layers:
                        print(f'WARNING: Specified fine tune head will result in bottleneck. Increasing layers to '
                              f'{self.fine_tune_input_layers}')
                        self.extra_class_layers[ind] = self.fine_tune_input_layer

            layers_to_replace = self.fine_tune_head + ['dropout']
            self.replace_layers_with_fc(layers_to_replace)

        if do_layer_freeze:
            self.model = freeze_layers(self.fine_tune_head, self.model)
            mlflow.log_param("Fine tune layers", self.fine_tune_head)
        else:
            mlflow.log_param("Fine tune layers", "All")

        param_total, param_trainable = get_model_param_num(self.model)
        mlflow.log_param("Model params", param_total)
        mlflow.log_param("Model params trainable", param_trainable)

        self.model.to(device)

    def replace_layers_with_fc(self, layers_to_replace):
        # Sanitize inputs in case layers_to_replace specifies weights and biases
        clear_layers(self.model, layers_to_replace)

        if self.extra_class_layers:
            classifier_layer_names = list()
            classifier_dropout_layer_names = list()
            extra_classifiers = nn.Sequential()

            layer_range = range(len(self.extra_class_layers))

            for layer_ind in layer_range:  # Skipped if range is empty
                if layer_ind == 0:
                    input_channels = self.fine_tune_input_layers
                else:
                    input_channels = self.extra_class_layers[layer_ind-1]

                classifier_layer_names += ['extra_class' + str(layer_ind)]
                extra_classifiers.add_module(name=classifier_layer_names[-1],
                                             module=nn.Linear(input_channels,
                                                              self.extra_class_layers[layer_ind]))
                classifier_dropout_layer_names += ['extra_class_dropout' + str(layer_ind)]
                extra_classifiers.add_module(name=classifier_dropout_layer_names[-1],
                                             module=nn.Dropout(p=self.fine_tune_dropout_rate))
                extra_classifiers.add_module(name=classifier_dropout_layer_names[-1],
                                             module=nn.Dropout(p=self.fine_tune_dropout_rate))
                extra_classifiers.add_module(name='extra_class_activation' + str(layer_ind),
                                             module=nn.GELU())

            classifier_layer_names += ['extra_class_final']
            extra_classifiers.add_module(name=classifier_layer_names[-1],
                                         module=nn.Linear(self.extra_class_layers[-1],
                                                          1))

            self.fine_tune_head = ['classifier.' + name for name in classifier_layer_names]
            self.model.classifier = extra_classifiers

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        # if self.extra_class_layers:
        #     x = self.model.extra_classifiers(x)
        return x

    def predict(self, input_ids, attention_mask, labels=None):
        self.model.eval()
        outputs = self.forward(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.logits > 0

    def on_train_epoch_start(self):
        self.train_loss = list()
        self.train_acc = list()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        preds_batch = tensor_to_numpy(outputs['logits'] > 0.5)
        train_acc_batch = accuracy_score(preds_batch, tensor_to_numpy(labels))
        self.train_loss.append(tensor_to_numpy(loss))
        self.train_acc.append(train_acc_batch)
        return loss

    def on_train_epoch_end(self):
        self.log('train_loss', average_round_metric(self.train_loss))
        self.log('train_acc', average_round_metric(self.train_acc))

    def on_validation_epoch_start(self):
        self.val_loss = list()
        self.val_acc = list()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        preds_batch = tensor_to_numpy(outputs['logits'] > 0.5)
        val_acc_batch = accuracy_score(preds_batch, tensor_to_numpy(labels))
        loss = outputs.loss
        self.val_loss.append(tensor_to_numpy(loss))
        self.val_acc.append(val_acc_batch)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', average_round_metric(self.val_loss))
        self.log('val_acc', average_round_metric(self.val_acc))

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, threshold=1e-3)
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_acc"}


def check_model_supported(model_name):
    if model_name not in supported_models:
        raise NotImplementedError(f"The model support for {model_name} has not been implemented.")


def tokenizer_setup(tokenizer_name):
    check_model_supported(tokenizer_name)
    if tokenizer_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(supported_models['distilbert'])
    elif tokenizer_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(supported_models['bert'])
    elif tokenizer_name.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(supported_models['roberta'])
    elif tokenizer_name.startswith('llama'):
        tokenizer = AutoTokenizer.from_pretrained(supported_models['llama'])
    else:
        raise NotImplementedError()
    return tokenizer


# def qc_requested_models_supported(model_names):
#     models_unsupported = list()
#     for model_name in model_names:
#         try:
#             model = FineTuneLLM(num_classes=1, model_name=model_name)
#         except RuntimeError:
#             models_unsupported.append(model_name)
#     if models_unsupported:
#         raise ValueError(f'The following models are not supported {models_unsupported}')


def model_setup(save_dir, num_classes, model_name='distilbert-base-uncased', do_layer_freeze=True,
                extra_class_layers=None):
    trainer = trainer_setup(model_name, save_dir)
    model = FineTuneLLMAsClassifier(model_name=model_name, num_classes=num_classes, do_layer_freeze=do_layer_freeze,
                                    extra_class_layers=extra_class_layers)
    return model, trainer
