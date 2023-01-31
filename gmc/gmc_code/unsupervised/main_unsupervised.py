import os
import torch
import sacred
import gmc_code.unsupervised.ingredients.exp_ingredients as sacred_exp
import gmc_code.unsupervised.ingredients.machine_ingredients as sacred_machine
import gmc_code.unsupervised.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.unsupervised.ingredients.stage_specific.down_ingredients as sacred_down
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.unsupervised.modules.trainers.model_trainer import ModelLearner
from gmc_code.unsupervised.modules.sacred_loggers import SacredLogger
from gmc_code.unsupervised.modules.callbacks import OnEndModelTrainingMHD, OnEndDownTrainingMHD
from gmc_code.unsupervised.utils.general_utils import (
    setup_dca_evaluation_trainer,
    setup_model,
    setup_data_module,
    load_model,
    setup_downstream_classifier,
    setup_downstream_classifier_trainer,
    load_down_model,
    setup_downstream_evaluator,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "GMC_unsupervised_experiments",
    ingredients=[sacred_machine.machine_ingredient, 
                 sacred_exp.exp_ingredient, 
                 sacred_model.model_ingredient, 
                 sacred_model.model_train_ingredient,
                 sacred_down.down_train_ingredient],
)

import numpy as np
import random
def seed_everything_my(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@ex.capture
def log_dir_path(folder, _config, _run):

    #!model_type = str(_config["experiment"]["model"])
    model_type = str(_config["model"]["model"])
    exp_name = str(_config["experiment"]["scenario"])

    return os.path.join(
        _config["machine"]["m_path"],
        "evaluation/",
        model_type + "_" + exp_name,
        f'log_{_config["experiment"]["seed"]}',
        folder,
    )


@ex.capture
def trained_model_dir_path(file, _config, _run):

    return os.path.join(
        _config["machine"]["m_path"],
        "evaluation/",
        _config["model"]["model"] + "_" + _config["experiment"]["scenario"],
        #!_config["experiment"]["model"] + "_" + _config["experiment"]["scenario"],
        f'log_{_config["experiment"]["seed"]}',
        #"trained_models/",
        "saved_models",
        file
    )


@ex.capture
def load_hyperparameters(_config, _run):

    exp_cfg = _config["experiment"]
    scenario_cfg = _config["experiment"]["scenario_config"]
    #model_cfg = _config["experiment"]["model_config"]
    model_cfg = _config["model"]

    return exp_cfg, scenario_cfg, model_cfg


@ex.capture
def train_model(_config, _run):

    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    #!model_train_cfg = _config["experiment"]["model_train_config"]
    model_train_cfg = _config["model_train"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)
    seed_everything_my(exp_cfg["seed"])

    # Init model
    model = setup_model(
        #!model=exp_cfg["model"],
        model=model_cfg["model"],
        scenario=exp_cfg["scenario"],
        scenario_config=scenario_cfg,
        model_config=model_cfg,
        use_label=exp_cfg['use_label'])

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg)

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
        use_label=exp_cfg['use_label'])

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Train
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        #!filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
        filename=f"{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
        + "{epoch:02d}",
        monitor="val_loss",
        every_n_epochs=model_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        #!f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
        f"{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    if exp_cfg["scenario"] == "mhd":
        end_callback = OnEndModelTrainingMHD()
    else:
        raise ValueError("Error")

    # TEST_limit_train_batches = 0.01
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=model_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        callbacks=[checkpoint_callback, end_callback])

    # Train
    trainer.fit(model_trainer, data_module)

@ex.capture
def dca_eval_model(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, _ = load_hyperparameters()
    dca_eval_cfg = _config["experiment"]["dca_evaluation_config"]

    # Set the seeds
    seed_everything(dca_eval_cfg["random_seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model = load_model(sacred_config=_config, model_file=model_file)

    # Init Trainer
    dca_trainer = setup_dca_evaluation_trainer(
        model=model,
        machine_path=_config["machine"]["m_path"],
        scenario=exp_cfg["scenario"],
        config=dca_eval_cfg,
    )

    # Init Data Module
    dca_data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=dca_eval_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("results_dca_evaluation"),
        logger=sacred_logger,
    )

    trainer.test(dca_trainer, dca_data_module)
    return


@ex.capture
def train_downstream_classifer(_config, _run):
    
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["down_train"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)
    seed_everything_my(exp_cfg["seed"])

    # Load model
    #!model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model_file = trained_model_dir_path(model_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Init downstream model
    down_model = setup_downstream_classifier(
        scenario=exp_cfg["scenario"], model_config=model_cfg
    )

    # Init Trainer
    down_trainer = setup_downstream_classifier_trainer(
        scenario=exp_cfg["scenario"],
        rep_model=rep_model,
        down_model=down_model,
        train_config=down_train_cfg,
        use_label=exp_cfg['use_label']
    )
    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Callbacks
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        #!filename=f"down_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
        filename=f"down_{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
        + "{epoch:02d}",
        monitor="val_loss",
        every_n_epochs=down_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        #!f"down_{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
        f"down_{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    if exp_cfg["scenario"] == "mhd":
        end_callback = OnEndDownTrainingMHD()
    else:
        raise ValueError("Error")

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=down_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        callbacks=[checkpoint_callback, end_callback])

    trainer.fit(down_trainer, data_module)


@ex.capture
def eval_downstream_classifier(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["down_train"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)
    seed_everything_my(exp_cfg["seed"])

    #! path 확인 제대로
    # Load model
    #!model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model_file = trained_model_dir_path(model_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Load downstream model
    #!down_model_file = trained_model_dir_path("down_" + exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    down_model_file = trained_model_dir_path("down_" + model_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    down_model = load_down_model(sacred_config=_config, down_model_file=down_model_file)

    # Init Trainer
    down_trainer = setup_downstream_evaluator(
        scenario=exp_cfg["scenario"],
        rep_model=rep_model,
        down_model=down_model,
        train_config=down_train_cfg,
        modalities=_config["experiment"]["evaluation_mods"],
        seed=exp_cfg['seed'],                                    #! added for output logging
        log_name=exp_cfg['log_name'],                            #! added for output logging
        use_label=exp_cfg['use_label'],
    )

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=down_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
    )

    trainer.test(down_trainer, data_module)


@ex.main
def main(_config, _run):

    # Run experiment
    if _config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_model()
    elif _config["experiment"]["stage"] == "evaluate_dca":
        os.makedirs(log_dir_path("results_dca_evaluation"), exist_ok=True)
        dca_eval_model()
    elif _config["experiment"]["stage"] == "train_downstream_classfier":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_downstream_classifer()
    elif _config["experiment"]["stage"] == "evaluate_downstream_classifier":
        os.makedirs(log_dir_path("results_down"), exist_ok=True)
        eval_downstream_classifier()
    else:
        raise ValueError(
            "[Unsupervised Experiment] Incorrect stage of pipeline selected: " + str(_config["experiment"]["stage"])
        )


if __name__ == "__main__":
    ex.run_commandline()