from typing import Union, Optional

import numpy as np

from datasets import load_metric
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import VisionEncoderDecoderModel

import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as pl
import mlflow

from utils.torch_utils import tensor_to_numpy, average_round_metric

# NB: Speed up processing for negligible loss of accuracy. Verify acceptable accuracy for a production use case
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.allow_tf32 = True


class FineTuneTrOCR(pl.LightningModule):
    def __init__(self, processor, model_name: str = 'microsoft/trocr-large-printed', device: str = 'cuda',
                 learning_rate: float = 5e-5):
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

        super(FineTuneTrOCR, self).__init__()

        self.processor = processor

        model = VisionEncoderDecoderModel.from_pretrained(model_name)  # "microsoft/trocr-base-stage1"

        self.model = model
        self.model.to(device)

        self.learning_rate = learning_rate

    def forward(self, batch):
        x = self.model(batch)
        return x

    def on_train_epoch_start(self):
        self.train_loss = list()
        self.train_acc = list()

    def training_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['labels_shop']
        outputs = self.model(image)
        loss = outputs.loss
        preds_batch = tensor_to_numpy(self.model.generate(batch["pixel_values"].to(self.device)))
        train_cer = self.compute_cer(pred_ids=preds_batch, label_ids=labels)
        self.train_loss.append(tensor_to_numpy(loss))
        self.train_acc.append(train_cer)
        return loss

    def on_train_epoch_end(self):
        self.log('train_loss', average_round_metric(self.train_loss))
        self.log('train_acc', average_round_metric(self.train_acc))

    def on_validation_epoch_start(self):
        self.val_loss = list()
        self.val_acc = list()

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        labels = batch['labels_shop']
        outputs = self.model(image)
        loss = outputs.loss
        preds_batch = tensor_to_numpy(self.model.generate(batch["pixel_values"].to(self.device)))
        val_cer = self.compute_cer(pred_ids=preds_batch, label_ids=labels)
        self.val_loss.append(tensor_to_numpy(loss))
        self.val_acc.append(val_cer)
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


def ocr_print(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.

    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    generated_text, score = ocr(image, processor, model, do_print=True)
    return generated_text, score


def ocr(image, processor, model, do_print=False):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.

    Returns:
        generated_text: the OCR'd text string.
    """
    pixel_values = processor(image, return_tensors='pt').pixel_values.to('cuda')
    return_dict = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
    generated_text = processor.batch_decode(return_dict['sequences'], skip_special_tokens=True)[0]

    if do_print:
        # print(return_dict['sequences'])
        # TODO: This way of interpreting scores does not correlate with good predictions. Replace
        max_score = np.max([tensor_to_numpy(torch.mean(score_tensor)) for score_tensor in return_dict['scores']])
        print(f"Score: {max_score}")
        print(generated_text)
    return generated_text, max_score
