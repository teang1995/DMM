from email.policy import strict
from pytorch_lightning.callbacks import Callback
from gmc_code.supervised_dmm.data_modules.class_dataset import *
from gmc_code.supervised_dmm.architectures.models.gmc import AffectGMC
from gmc_code.supervised_dmm.modules.trainers.dca_evaluation_trainer import DCAEvaluator


def setup_model(scenario, model_config):
    if scenario == "mosei" or scenario == 'mosi':
        return AffectGMC(
            name=model_config["model"],
            common_dim=model_config["common_dim"],
            latent_dim=model_config["latent_dim"],
            loss_type=model_config["loss_type"],
            scenario=scenario,
            #! DMM-related
            ldmm=model_config["ldmm"],
            sche_offset=model_config["sche_offset"],
            init_margin=model_config["init_margin"],
            in_norm=model_config['in_norm'],
            gmc=model_config['gmc'],
        )

    else:
        raise ValueError(
            "[Model Setup] Selected scenario not yet implemented for GMC Single model: "
            + str(scenario)
        )


def setup_data_module(scenario, experiment_config, scenario_config, train_config):
    if scenario == 'mosei' or scenario == 'mosi':
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


"""

Loading functions

"""


def load_model(sacred_config, model_file):

    model = setup_model(
        scenario=sacred_config["experiment"]["scenario"],
        model_config=sacred_config["model"]
    )

    checkpoint = torch.load(model_file)

    # if sacred_config["experiment"]["scenario"] == 'mosi' and sacred_config["experiment"]["model"][:3] == 'gmc':
    #     state_dict = delight_state_dict(checkpoint["state_dict"])
    #     model.load_state_dict(state_dict)
    # else:
    #     model.load_state_dict(checkpoint["state_dict"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    # Freeze model
    model.freeze()
    model.cuda()
    
    return model

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


def delight_state_dict(old_state_dict):
    new_state_dict = {}
    for key in old_state_dict:
        # Remove "model." from keys
        new_key = key[6:]
        new_state_dict[new_key] = old_state_dict[key]

    return new_state_dict


