from html.entities import name2codepoint
from pytorch_lightning import LightningModule
from gmc_code.unsupervised.architectures.models.gmc_networks import *
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

class GMC(LightningModule):
    def __init__(self, 
                 name, 
                 common_dim, 
                 latent_dim, 
                 loss_type="infonce",
                 mixup_type="no",
                 mixup_dist="dirichlet",
                 dirichlet_params=[2.0,2.0,2.0,2.0],
                 beta_params=[2.0,2.0]):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None

    # todo: if we use mmixup only, then our complete representation is ?
    # todo: choices = ['average'] ...only ?
    def encode(self, x, sample=False):
        # If we have complete observations
        if (None not in x) and (self.mixup_type != 'replace'):
            return self.encoder(self.processors[-1](x))
        else: #! complete obs with replace-type mixup or imcomplete obs
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations 
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def forward(self, x): #* x is batch of [mod1, mod2, mod3, (mod4)]
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            if x[processor_idx] is not None:
                mod_representations = self.encoder(
                    self.processors[processor_idx](x[processor_idx])
                )
                batch_representations.append(mod_representations)
            else:
                pass
        
        # Forward pass through the joint encoder
        #! if we replace GMC with mmixup
        if self.mixup_type == 'replace':
            return batch_representations
        #! else [no mixup, add]
        else:
            joint_representation = self.encoder(self.processors[-1](x)) #! 수정 필요 08.01 => 수정완료
            batch_representations.append(joint_representation)
            return batch_representations

    #! GMC with mmixup variants
    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        n_modals = len(batch_representations) - 1
        b_mult = 2
        #! -----------------------------------------------------------
        if self.mixup_type != 'no':
            mix_prop = self.dirichlet.sample().tolist()
            mix_rep = torch.zeros_like(batch_representations[0])
            
            for mod in range(n_modals):
                mix_rep += mix_prop[mod]*batch_representations[mod]
            batch_representations.append(mix_rep)

            if self.mixup_type == "replace": 
                mix_idx = -1
                b_mult = 2
            if self.mixup_type == "add": 
                mix_idx = -1
                joint_idx = -2
                b_mult = 3
        #! -----------------------------------------------------------

        for mod in range(n_modals):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            if self.mixup_type == 'add': #! [[mix ; modal] ; joint]
                out_joint_mod = torch.cat(
                    [out_joint_mod, batch_representations[-2]], dim=0
                )
            # [b_mult*B, b_mult*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [b_mult*B, b_mult*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(b_mult * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove b_mult*B diagonals and reshape to [b_mult*B, b_mult*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(b_mult * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            if self.mixup_type == 'add':
                pos_sim_joint_mod += torch.exp(
                    torch.sum(
                        batch_representations[-2] * batch_representations[mod], dim=-1
                    )
                    / temperature
                )
            # [b_mult*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod]*b_mult, dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict
    
    def infonce_with_joints_as_negatives(
        self, batch_representations, temperature, batch_size
    ):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / temperature
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
            torch.ones_like(sim_matrix_joints)
            - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(
            batch_size, -1
        )

        # compute loss - for each pair joints-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params):
        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return loss, tqdm_dict

    def validation_step(self, data, train_params):
        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        return tqdm_dict


#! add mixup type
class MhdGMC(GMC):
    '''
        * four modalities: [img, sound, traj, label]
    '''
    def __init__(self, 
                 name, 
                 common_dim, 
                 latent_dim, 
                 loss_type="infonce", 
                 mixup_type="no",
                 mixup_dist="dirichlet",
                 dirichlet_params=[2.0,2.0,2.0,2.0],
                 beta_params=[2.0,2.0],
                 use_label=True):
        super(MhdGMC, self).__init__(name, common_dim, latent_dim, loss_type, mixup_type, mixup_dist, dirichlet_params, beta_params)

        self.image_processor = MHDImageProcessor(common_dim=common_dim)
        self.sound_processor = MHDSoundProcessor(common_dim=common_dim)
        self.trajectory_processor = MHDTrajectoryProcessor(common_dim=common_dim)
        self.label_processor = MHDLabelProcessor(common_dim=common_dim)
        self.joint_processor = MHDJointProcessor(common_dim=common_dim, use_label=use_label)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.trajectory_processor,
            self.label_processor,
            self.joint_processor,
        ]
        self.encoder = MHDCommonEncoder(common_dim=common_dim, latent_dim=latent_dim)

        self.loss_type = loss_type
        self.mixup_type = mixup_type
        self.mixup_dist = mixup_dist
        self.use_label = use_label

        if name[:3] != 'gmc':
            if not use_label:
                if len(dirichlet_params) != 3:
                    raise ValueError
        self.dirichlet = Dirichlet(torch.tensor(dirichlet_params))
        self.beta = Beta(torch.tensor([beta_params[0]]), torch.tensor([beta_params[1]]))
        #mix_ratio_tensor = diric.sample()