import torch
from torch.nn import functional as F
from torch.optim import AdamW
import lightning as pl

from datasets import load_metric
from utils.torch_utils import tensor_to_numpy, average_round_metric


class AbstractiveQAFineTuner(pl.LightningModule):
    def __init__(self, model_name: str = 't5-base', device: str = 'cuda', learning_rate: float = 5e-5):
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

        self.train_loss = None
        self.train_acc_company = None
        self.train_acc_total = None

        self.val_loss = None
        self.val_acc_company = None
        self.val_acc_total = None

        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

        self.learning_rate = learning_rate

    def forward(self, batch):
        x = self.model(batch)
        return x

    def on_train_epoch_start(self):
        self.train_loss = list()
        self.train_acc_company = list()
        self.train_acc_total = list()

    def training_step(self, batch, batch_idx):
        loss_company, cer_company, loss_total, cer_total = self._run_inference_on_example(batch)
        loss = tensor_to_numpy(torch.mean(loss_company, loss_total))

        self.train_loss.append(loss)
        self.train_acc_company.append(cer_company)
        self.train_acc_total.append(cer_total)
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
        outputs = self.model.generate(input_ids=prompt, attention_mask=attention_mask, do_sample=False)
        preds = self.tokenizer.batch_decode(tensor_to_numpy(outputs))

        loss = F.cross_entropy(labels_tokens, outputs.logitd)
        cer = self.compute_cer(pred_ids=preds, label_ids=labels)
        return loss, cer

    def on_train_epoch_end(self):
        self.log('train_loss', average_round_metric(self.train_loss))
        self.log('train_acc', average_round_metric(self.train_acc_company))

    def on_validation_epoch_start(self):
        self.val_loss = list()
        self.val_acc_company = list()
        self.val_acc_total = list()

    def validation_step(self, batch, batch_idx):
        loss_company, cer_company, loss_total, cer_total = self._run_inference_on_example(batch)
        loss = tensor_to_numpy(torch.mean(loss_company, loss_total))

        self.val_loss.append(loss)
        self.val_acc_company.append(cer_company)
        self.val_acc_total.append(cer_total)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss', average_round_metric(self.val_loss))
        self.log('val_acc', average_round_metric(self.val_acc))

    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, threshold=1e-3)
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_acc"}

    def compute_cer(self, pred_ids, label_ids):
        cer_metric = load_metric("cer")  # Character error rate
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return cer


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
