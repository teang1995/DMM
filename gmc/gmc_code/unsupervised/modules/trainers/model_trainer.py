import torch
import torch.optim as optim
from collections import OrderedDict
from pytorch_lightning import LightningModule


class ModelLearner(LightningModule):
    def __init__(
        self, model, scenario, train_config, scenario_config, experiment_config, use_label=True
    ):
        super(ModelLearner, self).__init__()

        self.model = model
        self.scenario = scenario
        self.experiment_config = experiment_config
        self.train_config = train_config
        self.scenario_config = scenario_config
        self.use_label = use_label

    def configure_optimizers(self):
        optimiser = optim.Adam(
            self.model.parameters(), lr=self.train_config["learning_rate"], weight_decay=self.train_config["weight_decay"]
        )
        return optimiser

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        if self.scenario == 'mhd':
            #!data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            if self.use_label:
                data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            else:
                data = [batch[0], batch[1], batch[2], None]
        else:
            raise ValueError(
                "[Model Learner] Scenario not yet implemented: " + str(self.scenario)
            )
        loss, tqdm_dict = self.model.training_step(data, self.train_config)
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def training_epoch_end(self, outputs):
        log_keys = list(outputs[0]["log"].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx]["log"][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )

            # Add to sacred
            self.logger.log_metric(
                f"train_{log_key}", avg_batch_log, self.current_epoch
            )

    def validation_step(self, batch, batch_idx):

        if self.scenario == 'mhd':
            #!data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            if self.use_label:
                data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            else:
                data = [batch[0], batch[1], batch[2], None]
        else:
            raise ValueError(
                "[Model Learner] Scenario not yet implemented: " + str(self.scenario)
            )

        output_dict = self.model.validation_step(data, self.train_config)
        return output_dict

    def validation_epoch_end(self, outputs):
        log_keys = list(outputs[0].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )
            self.log(f"val_{log_key}", avg_batch_log, on_epoch=True, logger=False)
            self.logger.log_metric(f"val_{log_key}", avg_batch_log, self.current_epoch)

