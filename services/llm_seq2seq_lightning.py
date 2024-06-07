import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
import lightning as pl

from datasets import load_metric

from torch_utils import tensor_mean
from utils.torch_utils import tensor_to_numpy, average_round_metric


class AbstractiveQAFineTuner(pl.LightningModule):
    def __init__(self, model_name: str = 't5-base', device: str = 'cuda', learning_rate: float = 5e-5,
                 loss_metric='val_acc'):
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

        super(AbstractiveQAFineTuner, self).__init__()

        tokenizer, model = initialize_model(model_name)

        self.train_loss_total = None
        self.train_loss_company = None
        self.train_loss_overall = None
        self.train_cer_company = None
        self.train_cer_total = None

        self.val_loss_total = None
        self.val_loss_company = None
        self.val_loss_overall = None
        self.val_cer_company = None
        self.val_cer_total = None

        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

        self.learning_rate = learning_rate
        self.loss_metric = loss_metric
        self.cer = load_metric("cer")  # Character error rate

    def forward(self, batch):
        x = self.model(batch)
        return x

    def on_train_epoch_start(self):
        self.train_loss_total = list()
        self.train_loss_company = list()
        self.train_loss_overall = list()
        self.train_cer_total = list()
        self.train_cer_company = list()

    def training_step(self, batch, batch_idx):
        loss_company, cer_company, loss_total, cer_total = self._run_inference_on_example(batch)
        loss = tensor_mean(loss_company, loss_total)

        self.train_loss_total.append(tensor_to_numpy(loss_total))
        self.train_loss_company.append(tensor_to_numpy(loss_company))
        self.train_loss_overall.append(tensor_to_numpy(loss))
        self.train_cer_total.append(cer_total)
        self.train_cer_company.append(cer_company)
        # TODO: This looks hacky. Supposedly, the issue is actually caused by detaching tensors along the way,
        #  but which detatchment is the problem?
        loss.requires_grad = True
        return loss

    def _run_inference_on_example(self, batch):
        loss_company, cer_company = self._predict_on_prompt(batch, 'total')

        loss_total, cer_total = self._predict_on_prompt(batch, 'company')
        return loss_company, cer_company, loss_total, cer_total

    def _predict_on_prompt(self, batch, label_name):
        prompt = batch['prompt_' + label_name]
        attention_mask = batch['prompt_' + label_name + "_att"]
        labels = batch[label_name]
        labels_tokens = batch["tokens_"+label_name]
        outputs = self.model.generate(input_ids=prompt,
                                      attention_mask=attention_mask,
                                      max_length=len(labels_tokens),
                                      do_sample=False)
        preds = self.tokenizer.batch_decode(tensor_to_numpy(outputs))

        label_tokens_for_loss_calc = torch.stack(labels_tokens, dim=1).type(torch.FloatTensor).to('cuda')
        output_tokens_for_loss_calc = outputs.type(torch.FloatTensor).to('cuda')
        output_tokens_for_loss_calc = F.pad(output_tokens_for_loss_calc, (1,
                                                                          label_tokens_for_loss_calc.shape[1] -
                                                                          output_tokens_for_loss_calc.shape[1] - 1))
        mask = output_tokens_for_loss_calc > 0
        loss = F.cross_entropy(label_tokens_for_loss_calc[mask], output_tokens_for_loss_calc[mask])
        cer = self.cer.compute(predictions=preds, references=labels)
        return loss, cer

    def on_train_epoch_end(self):
        self.log('train_loss_total', average_round_metric(self.train_loss_total))
        self.log('train_loss_company', average_round_metric(self.train_loss_company))
        self.log('train_loss_overall', average_round_metric(self.train_loss_overall))
        self.log('train_cer_company', average_round_metric(self.train_cer_company))
        self.log('train_cer_total', average_round_metric(self.train_cer_total))

    def on_validation_epoch_start(self):
        self.val_loss_total = list()
        self.val_loss_company = list()
        self.val_loss_overall = list()
        self.val_cer_total = list()
        self.val_cer_company = list()

    def validation_step(self, batch, batch_idx):
        loss_company, cer_company, loss_total, cer_total = self._run_inference_on_example(batch)
        loss = tensor_mean(loss_company, loss_total)

        self.val_loss_total.append(tensor_to_numpy(loss_total))
        self.val_loss_company.append(tensor_to_numpy(loss_company))
        self.val_loss_overall.append(tensor_to_numpy(loss))
        self.val_cer_total.append(cer_total)
        self.val_cer_company.append(cer_company)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss_total', average_round_metric(self.val_loss_total))
        self.log('val_loss_company', average_round_metric(self.val_loss_company))
        self.log('val_loss_overall', average_round_metric(self.val_loss_overall))
        self.log('val_cer_total', average_round_metric(self.val_cer_total))
        self.log('val_cer_company', average_round_metric(self.val_cer_company))

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, threshold=1e-3)
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": self.loss_metric}


def initialize_model(model_name):
    if model_name.startswith('t5-base'):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # elif 'pegasus' in model_name:
    #     from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    #     tokenizer = PegasusTokenizer.from_pretrained(model_name)
    #     model = PegasusForConditionalGeneration.from_pretrained(model_name).to('cuda')
    else:
        raise NotImplementedError('')

    return tokenizer, model
