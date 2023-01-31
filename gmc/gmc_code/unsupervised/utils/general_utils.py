from gmc_code.unsupervised.architectures.models.gmc import MhdGMC
from gmc_code.unsupervised.architectures.downstream.classifiers import ClassifierMNIST
from gmc_code.unsupervised.modules.trainers.classification_trainer import ClassifierLearner
from gmc_code.unsupervised.modules.trainers.dca_evaluation_trainer import DCAEvaluator
from gmc_code.unsupervised.data_modules.class_dataset import *


def setup_model(scenario, model, model_config, scenario_config, data_module=None, use_label=True):
    #!if model in ["gmc", "mmix"]:
    if scenario == "mhd":
        return MhdGMC(
            name=model_config["model"],
            common_dim=model_config["common_dim"],
            latent_dim=model_config["latent_dim"],
            loss_type=model_config["loss_type"],
            #! mmixup-related
            mixup_type=model_config["mixup_type"],
            mixup_dist=model_config["mixup_dist"],
            dirichlet_params=model_config["dirichlet_params"],
            beta_params=model_config["beta_params"],
            use_label=use_label
        )

    else:
        raise ValueError(
            "[Model Setup] Selected scenario not yet implemented for GMC model: "
            + str(scenario)
        )

    # else:
    #     raise ValueError(
    #         "[Model Setup] Selected model not yet implemented: " + str(model)
    #     )


def setup_data_module(scenario, experiment_config, scenario_config, train_config):
    if scenario == "mhd":
        if experiment_config["stage"] == "evaluate_dca":
            return DCADataModule(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                data_config=train_config,
            )
        else:
            return ClassificationDataModule(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                data_config=train_config,
            )
    else:
        raise ValueError(
            "[Data Module Setup] Selected Module not yet implemented: " + str(scenario)
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )


def setup_downstream_classifier(scenario, model_config):
    if scenario == 'mhd':  # Classification
        return ClassifierMNIST(latent_dim=model_config["latent_dim"])
    else:
        raise ValueError(
            "[Down Model Setup] Selected scenario not yet implemented: " + str(scenario)
        )


def setup_downstream_classifier_trainer(scenario, rep_model, down_model, train_config, use_label=True):
    if scenario == 'mhd':  # Classification
        return ClassifierLearner(
            model=rep_model, scenario=scenario, classifier=down_model, train_config=train_config,
            use_label=use_label
        )

    else:
        raise ValueError(
            "[Down Classifier Trainer Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


def setup_downstream_evaluator(
    scenario, rep_model, down_model, train_config, modalities, seed, log_name, use_label=True
):
    if scenario == 'mhd':  # Classification
        return ClassifierLearner(
            model=rep_model,
            scenario=scenario,
            classifier=down_model,
            train_config=train_config,
            modalities=modalities,
            seed=seed,
            log_name=log_name,
            use_label=use_label
        )
    else:
        raise ValueError(
            "[Down Evaluator Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


"""

Loading functions

"""



def load_model(sacred_config, model_file):

    model = setup_model(
        scenario=sacred_config["experiment"]["scenario"],
        model=sacred_config["model"]["model"],
        model_config=sacred_config["model"],
        scenario_config=sacred_config["experiment"]["scenario_config"],
        use_label=sacred_config["experiment"]["use_label"],
    )

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()

    return model


def load_down_model(sacred_config, down_model_file):


    down_model = setup_downstream_classifier(
        scenario=sacred_config["experiment"]["scenario"],
        model_config=sacred_config["model"],
    )

    checkpoint = torch.load(down_model_file)
    down_model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    down_model.freeze()
    down_model.eval()

    return down_model


"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )

