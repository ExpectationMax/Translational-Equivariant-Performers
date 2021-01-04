import pytorch_lightning as pl


class TbWithMetricsLogger(pl.loggers.TensorBoardLogger):
    def __init__(self, save_dir, initial_values, **kwargs):
        super().__init__(save_dir, default_hp_metric=False, **kwargs)
        self.hparams_saved = False
        self.initial_values = initial_values

    @pl.utilities.rank_zero_only
    def log_hyperparams(self, params):
        print('Called log hparams')
        # Somehow hyperparameters are saved when a model is simply restored,
        # catch that here so we don't add an incorrect value when restoring.
        if self.hparams_saved:
            return
        super().log_hyperparams(
            params,
            self.initial_values
        )
        self.hparams_saved = True
