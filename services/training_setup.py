import lightning as pl
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor


def trainer_setup(model_name, save_dir, loss_metric="val_acc"):
    model_name_clean = model_name.split('\\')[-1]
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename=model_name_clean + "-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=1,
                                          monitor=loss_metric)
    early_stop_callback = EarlyStopping(monitor=loss_metric, min_delta=0.0001, patience=5, verbose=False,
                                        mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir)
    trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
                         logger=tb_logger, log_every_n_steps=50)
    return trainer
