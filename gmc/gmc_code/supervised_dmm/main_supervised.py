import os
import torch
import sacred
import gmc_code.supervised_dmm.ingredients.exp_ingredients as sacred_exp
import gmc_code.supervised_dmm.ingredients.machine_ingredients as sacred_machine
import gmc_code.supervised_dmm.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.supervised_dmm.ingredients.stage_specific.down_ingredients as sacred_down
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.supervised_dmm.modules.trainers.model_trainer import ModelLearner
from gmc_code.supervised_dmm.modules.trainers.model_evaluation import ModelEvaluation
from gmc_code.supervised_dmm.modules.callbacks import OnEndModelTrainingMosi, OnEndModelTrainingMosei
from gmc_code.supervised_dmm.modules.sacred_loggers import SacredLogger

from gmc_code.supervised_dmm.utils.general_utils import (
    setup_model,
    setup_data_module,
    load_model,
    setup_dca_evaluation_trainer,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "GMC_unsupervised_dmm_experiments",
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
        #"trained_models/",
        "evaluation/",
        _config["model"]["model"] + "_" + _config["experiment"]["scenario"],
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
        exp_cfg['scenario'],
        model_config=model_cfg,
    )

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

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
    if exp_cfg["scenario"] == "mosei":
        end_callback = OnEndModelTrainingMosei(model_path=log_dir_path("checkpoints"),
                                               model_filename=f"{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth")

    elif exp_cfg["scenario"] == "mosi":
        end_callback = OnEndModelTrainingMosi(model_path=log_dir_path("checkpoints"),
                                              model_filename=f"{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth")

    else:
        raise ValueError("Error")

    # TEST_limit_train_batches = 0.01
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=model_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        callbacks=[checkpoint_callback, end_callback],
        gradient_clip_val=0.8,
    )

    # Train
    trainer.fit(model_trainer, data_module)

    # sacred_logger.log_artifact(
    #     name=f"{exp_cfg['model']}_{exp_cfg['scenario']}_model.pth.tar",
    #     filepath=os.path.join(log_dir_path("checkpoints"),
    #                           f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth"))
    sacred_logger.log_artifact(
        name=f"{model_cfg['model']}_{exp_cfg['scenario']}_model.pth.tar",
        filepath=os.path.join(log_dir_path("checkpoints"),
                              f"{model_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth"))


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
def evaluate(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    #down_train_cfg = _config["experiment"]["down_train_config"]
    down_train_cfg = _config["down_train"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)
    seed_everything_my(exp_cfg["seed"])

    # Load model
    #!model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model_file = trained_model_dir_path(model_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    encoder_model = load_model(sacred_config=_config, model_file=model_file)

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Evaluator
    affect_evaluator = ModelEvaluation(model_name=exp_cfg['model'],
                                        model=encoder_model,
                                        scenario=exp_cfg['scenario'],
                                        sacred_logger=sacred_logger,
                                        test_loader=test_dataloader,
                                        modalities=_config["experiment"]["evaluation_mods"],
                                        seed=exp_cfg["seed"])

    affect_evaluator.evaluate()


@ex.capture
def embedding_analysis(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["down_train"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)
    seed_everything_my(exp_cfg["seed"])

    # Load model
    model_file = trained_model_dir_path(model_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    encoder_model = load_model(sacred_config=_config, model_file=model_file)

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    #! ----------------------------------------- 
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from umap import UMAP
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.nn.functional import normalize
    import pdb

    obs_mod = _config["experiment"]["evaluation_mods"]

    if obs_mod == [0]: mod_n = 'T'
    if obs_mod == [1]: mod_n = 'A'
    if obs_mod == [2]: mod_n = 'V'
    if obs_mod == [0,1]: mod_n = 'TA'
    if obs_mod == [0,2]: mod_n = 'TV'
    if obs_mod == [1,2]: mod_n = 'AV'
    if obs_mod == [0,1,2]: mod_n = 'TAV'

    affect_evaluator = ModelEvaluation(model_name=exp_cfg['model'],
                                        model=encoder_model,
                                        scenario=exp_cfg['scenario'],
                                        sacred_logger=sacred_logger,
                                        test_loader=test_dataloader,
                                        modalities=obs_mod,
                                        seed=exp_cfg["seed"])

    # get embedding
    partial, target = affect_evaluator.get_emb(mod=obs_mod)
    joint, _ = affect_evaluator.get_emb(mod=[0,1,2])
    partial = normalize(partial)
    joint = normalize(joint)

    partial = partial.cpu()
    joint = joint.cpu()
    pos_align = (partial-joint).norm(dim=1).pow(2)
    diff_tensor = (partial.unsqueeze(1) - joint.unsqueeze(0)).view(partial.shape[0], partial.shape[0], partial.shape[1])  # N*N*D element-wise diff
    diff_tensor = diff_tensor.norm(dim=2).pow(2)         # N*N diffence matrix(aggregated by feature dim)
    diff_tensor.fill_diagonal_(10000.0)                      # eliminate diagonal(pos.pair) elements
    
    neg_align = diff_tensor.min(dim=1).values
    s_align = -(pos_align - neg_align).mean()

    s_punif = torch.pdist(partial.float(),p=2).pow(2)
    s_punif = s_punif.mul(-2).exp().mean().log()
    s_junif = torch.pdist(joint.float(),p=2).pow(2)
    s_junif = s_junif.mul(-2).exp().mean().log()

    # intersim = []
    # intrasim = []
    # target = target.reshape(-1)
    # for l in target.unique():
    #     #pdb.set_trace()
    #     group_l = partial[target == l]
    #     # inter sim
    #     intsimmat = group_l @ group_l.T
    #     intsimmat = torch.tril(intsimmat, -1)
    #     intersim.append(intsimmat[abs(intsimmat) > 0].mean().item())
    #     for j in range(group_l.shape[0]):
    #         # intra sim
    #         intrasim.append((group_l[j] @ partial[target != l].T).mean().item())
    # s_pinter, s_pintra = np.mean(intersim), np.mean(intrasim)

    # intersim = []
    # intrasim = []
    # for l in target.unique():
    #     group_l = joint[target == l]
    #     # inter sim
    #     intsimmat = group_l @ group_l.T
    #     intsimmat = torch.tril(intsimmat, -1)
    #     intersim.append(intsimmat[abs(intsimmat) > 0].mean().item())
    #     for j in range(group_l.shape[0]):
    #         # intra sim
    #         intrasim.append((group_l[j] @ partial[target != l].T).mean().item())
    # s_jinter, s_jintra = np.mean(intersim), np.mean(intrasim)


    f = open(os.path.join(log_dir_path("results_vis"), 'scores.txt'), 'a')
    print(f'mod: {mod_n} / alignment score: {s_align}' , file=f)
    print(f'mod: {mod_n} / uniformity score: {s_punif}' , file=f)
    print(f'mod: {mod_n} / uniformity score (j): {s_junif}' , file=f)
    f.close()

    # embedding visualization
    vis_flag = exp_cfg['vis']
    if vis_flag == 'tsne':
        N_sample = partial.size(0)
        vis_tensor = torch.cat([partial, joint], dim=0); del partial; del joint
        vis_array = np.array(vis_tensor.cpu()); del vis_tensor
        vis_emb = TSNE(n_components=2, random_state=exp_cfg["seed"]).fit_transform(vis_array); del vis_array
    elif vis_flag == 'umap':
        N_sample = partial.size(0)
        vis_tensor = torch.cat([partial, joint], dim=0); del partial; del joint
        vis_array = np.array(vis_tensor.cpu()); del vis_tensor
        vis_emb = UMAP(n_components=2, random_state=exp_cfg["seed"]).fit_transform(vis_array); del vis_array
    else:
        import sys
        sys.exit()


    dum0 = np.zeros((N_sample, 1))
    dum1 = np.ones((N_sample, 1))
    dum = np.concatenate((dum0, dum1), axis=0)

    vis_data = np.concatenate((vis_emb, dum), axis=1)
    vis_df = pd.DataFrame(vis_data, columns=['dim1','dim2','Modality'])
    vis_df.loc[vis_df['Modality'] == 0, 'Modality'] = mod_n
    vis_df.loc[vis_df['Modality'] == 1, 'Modality'] = 'joint'

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=vis_df, x='dim1', y='dim2', hue='Modality', palette=['#FF3366', '#33CC99'], s=15, alpha=0.6)
    plt.legend(title=r"$Modality$", title_fontsize=12, fontsize=9)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig(f'{log_dir_path("results_vis")}/{mod_n}_{vis_flag}.png', dpi=285)
    plt.clf()
    #! -----------------------------------------


@ex.main
def main(_config, _run):
    obs_mod = _config["experiment"]["evaluation_mods"]
    if obs_mod == [0]: mod_n = 'T'
    if obs_mod == [1]: mod_n = 'A'
    if obs_mod == [2]: mod_n = 'V'
    if obs_mod == [0,1]: mod_n = 'TA'
    if obs_mod == [0,2]: mod_n = 'TV'
    if obs_mod == [1,2]: mod_n = 'AV'
    if obs_mod == [0,1,2]: mod_n = 'TAV'

    # Run experiment
    if _config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_model()
    elif _config["experiment"]["stage"] == "evaluate_dca":
        os.makedirs(log_dir_path("results_dca_evaluation"), exist_ok=True)
        dca_eval_model()
    elif _config["experiment"]["stage"] == "evaluate_downstream_classifier":
        os.makedirs(log_dir_path("results_down"), exist_ok=True)
        evaluate()
    elif _config["experiment"]["stage"] == "evaluate_visualization":
        os.makedirs(log_dir_path("results_vis"), exist_ok=True)
        embedding_analysis()
    else:
        raise ValueError(
            "[supervised_dmm Experiment] Incorrect stage of pipeline selected: "
            + str(_config["experiment"]["stage"])
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    ex.run_commandline()