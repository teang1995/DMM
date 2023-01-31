from gmc_code.supervised.architectures.models.gmc_networks import *
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

class SuperGMC(LightningModule):
    def __init__(self, 
                 name, 
                 common_dim, 
                 latent_dim, 
                 loss_type="infonce",
                 mixup_type="no",
                 mixup_dist="dirichlet",
                 dirichlet_params=[1.0,1.0,1.0],
                 beta_params=[1.0,1.0],
                 infer_mixed=False):
        super(SuperGMC, self).__init__()

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
        self.proj1 = None
        self.proj2 = None
        self.classifier = None
        self.criterion = nn.L1Loss()


    def encode(self, x, sample=False):

        # If we have complete observations
        #if (None not in x) and (not self.infer_mixed):
        if None not in x:
            joint_representation = self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))
            # Take the average of the latent representations
            joint_representation = torch.stack(latent_representations, dim=0).mean(0)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
        output += joint_representation

        return self.classifier(output)

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
        output += joint_representation

        return self.classifier(output), batch_representations

    def super_gmc_loss(self, prediction, target, batch_representations, temperature, batch_size):
        joint_mod_loss_sum  = 0  
        n_modals = len(batch_representations) - 1

        for mod in range(n_modals):
            # Negative pairs
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod
        
        # #! contrastive learning with M3 mixup
        # if self.mixup_type != 'no':
        #     #! dirichlet mixed representation
        #     mix_prop = self.dirichlet.sample().tolist()
        #     mix_rep = torch.zeros_like(batch_representations[0])
            
        #     for mod in range(n_modals):
        #         mix_rep += mix_prop[mod]*batch_representations[mod]
        #     #batch_representations.append(mix_rep)
                
        #         # if use V, L mix
        #         lamb = self.beta.sample()
        #         anc =  lamb * batch_representations[mod] + (1-lamb)*torch.flip(batch_representations[mod], dims=[0])

        #     #! VL mix

        #     #! M2 mix
            
        supervised_loss = self.criterion(prediction, target)

        loss = torch.mean(joint_mod_loss_sum + supervised_loss)
        # loss = torch.mean(supervised_loss)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict


    def training_step(self, data, target_data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, temperature, batch_size)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, temperature, batch_size)
        return tqdm_dict



#! add mixup type
# Affect
class AffectGMC(SuperGMC):
    def __init__(self, 
                 name, 
                 common_dim, 
                 latent_dim, 
                 loss_type="infonce",
                 scenario='mosei',
                 mixup_type="no",
                 mixup_dist="dirichlet",
                 dirichlet_params=[1.0,1.0,1.0],
                 beta_params=[1.0,1.0],
                 infer_mixed=False):
        super(AffectGMC, self).__init__(name, common_dim, latent_dim, loss_type)

        if scenario == 'mosei':
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=74, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=35, hidden_dim=30, latent_dim=latent_dim, timestep=50)
        else:
            self.language_processor = AffectGRUEncoder(input_dim=300, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.audio_processor = AffectGRUEncoder(input_dim=5, hidden_dim=30, latent_dim=latent_dim, timestep=50)
            self.vision_processor = AffectGRUEncoder(input_dim=20, hidden_dim=30, latent_dim=latent_dim, timestep=50)

        self.joint_processor = AffectJointProcessor(latent_dim, scenario)

        self.processors = [
            self.language_processor,
            self.audio_processor,
            self.vision_processor,
            self.joint_processor
        ]
        self.encoder = AffectEncoder(common_dim=common_dim, latent_dim=latent_dim)

        self.loss_type = loss_type
        self.mixup_type = mixup_type
        self.mixup_dist = mixup_dist
        self.infer_mixed = infer_mixed

        self.dirichlet = Dirichlet(torch.tensor(dirichlet_params))
        self.beta = Beta(torch.tensor([beta_params[0]]), torch.tensor([beta_params[1]]))

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)