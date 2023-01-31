from gmc_code.supervised_mix.architectures.models.gmc_networks import *
from gmc_code.supervised_mix.architectures.models.loss import *
import pdb
import random
import numpy as np

def sph_inter(a,b,s):
    #s = s.to(a.device)
    theta = torch.acos( (a*b).sum(dim=[1] )).view(a.shape[0],1)
    n1 = torch.sin(s*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-s)*theta)/torch.sin(theta)*b
    return n1+n2

class SuperGMC(LightningModule):
    def __init__(self, 
                 name, 
                 common_dim, 
                 latent_dim, 
                 loss_type="infonce",
                 scenario='mosei',
                 m2mix_type='pos',
                 beta_param=1.0,
                 single_mix=0.0,
                 all_mix=0.0,
                 multi_mix=0.0,
                 mix_schedule=0,
                 tot_ep=40
                 ):
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


    def encode(self, x, sample=False, out_z=False):

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

        if out_z:
            return joint_representation
        else:
            return self.classifier(output)

    def forward(self, x):
        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            # mod_representations = mod_representations / mod_representations.norm(dim=-1, keepdim=True)
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        joint_representation = self.encoder(self.processors[-1](x))
        # joint_representation = joint_representation / joint_representation.norm(dim=-1, keepdim=True)
        batch_representations.append(joint_representation)

        # Forward classifier
        output = self.proj2(F.dropout(F.relu(self.proj1(joint_representation)), p=0.0, training=self.training))
        output += joint_representation

        return self.classifier(output), batch_representations

    def super_gmc_loss(self, prediction, target, batch_representations, temperature, batch_size, ep):
        joint_mod_loss_sum  = 0  
        loss_sm = 0
        loss_dm = 0
        loss_mm_sum = 0
        n_modals = len(batch_representations) - 1


        targets_orig    = torch.eye(batch_size).to(batch_representations[-1].device) # Identity Matrix
        I       = targets_orig
        I_R     = torch.flip(I,dims=[0])
        I_D     = 1-I

        def write_original_neg(current_neg,original_neg):
            cross_I   = I + I_R
            cross_I_D = 1 - cross_I
            return current_neg*cross_I + original_neg*cross_I_D

        #! geodesical mixup between two random modality
        #pdb.set_trace()
        lamb = torch.Tensor([random.betavariate(self.beta_param,self.beta_param)]).to(batch_representations[-1].device).detach()
        # lamb = np.random.uniform(0.1, 0.9)
        mod1, mod2 = random.sample([i for i in range(3)],2) # currently, I specify the number of modality to 3 for ease
        mix_rep = sph_inter(batch_representations[mod1], batch_representations[mod2], lamb)
        mix_rep = mix_rep / mix_rep.norm(dim=-1, keepdim=True)

        for mod in range(n_modals):
            # Negative pairs
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod_tot = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod_tot)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod_tot.device)
            ).bool()

            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod_tot.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.clamp(torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1)/ temperature, max=10)
                # torch.sum(batch_representations[-1] * batch_representations[mod], dim=-1)/ temperature
            )
            
            # expand to [2*B] for matching with Negatives 
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)

            # InfoNCE of GMC
            loss_joint_mod = -torch.log(pos_sim_joint_mod / (sim_matrix_joint_mod.sum(dim=-1) + 1e-6))

            joint_mod_loss_sum += loss_joint_mod

            #! ---------------------------------------------------------------------------
            if ep >= (self.mix_schedule * self.tot_ep):
                origin_sim = batch_representations[mod] @ batch_representations[-1].T
                #! V, L mix (=> uni / joint separated mix)
                #pdb.set_trace()
                if self.single_mix != 0:
                    # mixed unimod | origin joint
                    #! uniform으로 바꿔서 u(0.1 0.9)
                    # lamb = np.random.uniform(0.1, 0.9)
                    lamb = torch.Tensor([random.betavariate(self.beta_param,self.beta_param)]).to(sim_matrix_joint_mod.device).detach()
                    single_mix1_r = sph_inter(batch_representations[mod], torch.flip(batch_representations[mod], dims=[0]), lamb)
                    # single_mix1_r = single_mix1_r / single_mix1_r.norm(dim=-1, keepdim=True)
                    
                    single_mix_sim_matrix1 = single_mix1_r @ batch_representations[-1].T / temperature
                    # loss_sm += calc_mix_loss(single_mix_sim_matrix1, lamb)
                    single_mix_logit1 = write_original_neg(single_mix_sim_matrix1, origin_sim)
                    loss_sm += calc_mix_loss(single_mix_logit1, lamb)

                    # origin unimod | mixed joint
                    # lamb = np.random.uniform(0.1, 0.9)
                    lamb = torch.Tensor([random.betavariate(self.beta_param,self.beta_param)]).to(sim_matrix_joint_mod.device).detach()
                    single_mix2_r = sph_inter(batch_representations[-1], torch.flip(batch_representations[-1], dims=[0]), lamb)
                    # single_mix2_r = single_mix2_r / single_mix2_r.norm(dim=-1, keepdim=True)
                    
                    single_mix_sim_matrix2 = single_mix2_r @ batch_representations[mod].T / temperature
                    # loss_sm += calc_mix_loss(single_mix_sim_matrix2, lamb)
                    single_mix_logit2 = write_original_neg(single_mix_sim_matrix2, origin_sim)
                    loss_sm += calc_mix_loss(single_mix_logit2, lamb)
                    
                    # if (torch.isnan(single_mix_r)).sum().item() + (torch.isnan(batch_representations[mod])).sum().item() + ((torch.isnan(batch_representations[-1]))).sum().item() > 0:
                    #     print(f'lamb: {lamb}')
                    #     print(f'single_mix_r shape: {single_mix_r.shape}')
                    #     print(f'single_mix_r mean: {single_mix_r.mean()}')
                    #     print(f'single_mix_r: {single_mix_r}')
                    #     print(f'rep shape: {batch_representations[mod].shape}')
                    #     print(f'rep mean: {batch_representations[mod].mean()}')
                    #     print(f'rep: {batch_representations[mod]}')
                    #     print(f'joint rep: {batch_representations[-1]}')
                    #     print(f'single_mix_pos2 shape: {single_mix_pos2.shape}')
                    #     print(f'single_mix_pos2 mean: {single_mix_pos2.mean()}')
                    #     print(f'single_mix_pos2: {single_mix_pos2}')
                    #     print(f'single_mix_pos1 shape: {single_mix_pos1.shape}')
                    #     print(f'single_mix_pos1 mean: {single_mix_pos1.mean()}')
                    #     print(f'single_mix_pos1: {single_mix_pos1}')
                    #     print(f'sim_matrix_joint_mod shape: {sim_matrix_joint_mod.shape}')
                    #     print(f'sim_matrix_joint_mod: {sim_matrix_joint_mod}')
                    #     break
                        
                #! VL mix (todo: add flag)
                if self.all_mix != 0:
                    #if (batch_size % 2) == 0:
                    #pdb.set_trace()
                    # lamb = np.random.uniform(0.1, 0.9)
                    lamb = torch.Tensor([random.betavariate(self.beta_param,self.beta_param)]).to(sim_matrix_joint_mod.device).detach()
                    uni_mix_r = sph_inter(batch_representations[mod], torch.flip(batch_representations[mod], dims=[0]), lamb)
                    # uni_mix_r = uni_mix_r / uni_mix_r.norm(dim=-1, keepdim=True)
                    joint_mix_r = sph_inter(batch_representations[-1], torch.flip(batch_representations[-1], dims=[0]), lamb)
                    # joint_mix_r = joint_mix_r / joint_mix_r.norm(dim=-1, keepdim=True)
                    
                    vl_mix_sim = uni_mix_r @ joint_mix_r.T
                    vl_mix_logit = write_original_neg(vl_mix_sim, origin_sim)
                    loss_dm += clip_loss(vl_mix_logit, I)

                    # if (torch.isnan(single_mix_r)).sum().item() + (torch.isnan(joint_mix_r)).sum().item() + ((torch.isnan(vl_mix_pos))).sum().item() > 0:
                    #     print(f'lamb2: {lamb2}')
                    #     print(f'single_mix_r shape: {single_mix_r.shape}')
                    #     print(f'single_mix_r mean: {single_mix_r.mean()}')
                    #     print(f'single_mix_r: {single_mix_r}')
                    #     print(f'rep shape: {batch_representations[mod].shape}')
                    #     print(f'rep mean: {batch_representations[mod].mean()}')
                    #     print(f'rep: {batch_representations[mod]}')
                    #     print(f'joint rep: {batch_representations[-1]}')
                    #     print(f'joint_mix_r shape: {joint_mix_r.shape}')
                    #     print(f'joint_mix_r mean: {joint_mix_r.mean()}')
                    #     print(f'joint_mix_r: {joint_mix_r}')
                    #     print(f'vl_mix_pos shape: {vl_mix_pos.shape}')
                    #     print(f'vl_mix_pos mean: {vl_mix_pos.mean()}')
                    #     print(f'vl_mix_pos: {vl_mix_pos}')
                    #     print(f'sim_matrix_joint_mod shape: {sim_matrix_joint_mod.shape}')
                    #     print(f'sim_matrix_joint_mod: {sim_matrix_joint_mod}')
                    #     break

                #! M2 mix (todo: pos or pos+neg)
                if self.multi_mix != 0:
                    # neg
                    out_mixed_mod = torch.cat([mix_rep, batch_representations[-1]], dim=0)
                    sim_matrix_mixed_mod = torch.exp(torch.mm(out_mixed_mod, out_mixed_mod.t().contiguous()) / temperature)
                    
                    mask_joint_mod = (
                        torch.ones_like(sim_matrix_mixed_mod)
                        - torch.eye(2 * batch_size, device=sim_matrix_mixed_mod.device)
                    ).bool()
                    sim_matrix_mixed_mod = sim_matrix_mixed_mod.masked_select(mask_joint_mod).view(2 * batch_size, -1)

                    # pos
                    pos_sim_mixed_joint = torch.exp(torch.sum(mix_rep * batch_representations[-1], dim=-1) / temperature)
                    pos_sim_mixed_joint = torch.cat([pos_sim_mixed_joint, pos_sim_mixed_joint], dim=0)   # align(rand 2 mixed mod, joint)

                    #pdb.set_trace()
                    if self.m2mix_type == 'pos': 
                        loss_mm = -torch.log(pos_sim_mixed_joint / sim_matrix_joint_mod.sum(dim=-1))     # mixed pos, origin neg
                        loss_mm_sum += loss_mm
                    elif self.m2mix_type == 'both':
                        loss_mm = -torch.log(pos_sim_mixed_joint / sim_matrix_mixed_mod.sum(dim=-1))     # mixed pos, mixed neg (consider only one time)
                        loss_mm_sum = loss_mm
                    else: # neg only
                        loss_mm = -torch.log(pos_sim_joint_mod / sim_matrix_mixed_mod.sum(dim=-1))       # origin pos, mixed neg
                        loss_mm_sum += loss_mm
                #! ---------------------------------------------------------------------------    
            
        supervised_loss = self.criterion(prediction, target)
        if ep >= (self.mix_schedule * self.tot_ep):
            loss = torch.mean(joint_mod_loss_sum + supervised_loss) + (self.multi_mix*loss_mm_sum).mean()
            # loss = torch.mean(joint_mod_loss_sum + supervised_loss) + (self.single_mix*loss_sm).mean() + (self.all_mix*loss_dm).mean() + (self.multi_mix*loss_mm_sum).mean()
        else:
            loss = torch.mean(joint_mod_loss_sum + supervised_loss)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict


    def training_step(self, data, target_data, train_params, ep):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive + supervised loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, temperature, batch_size, ep)

        return loss, tqdm_dict

    def validation_step(self, data, target_data, train_params, ep=0):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        output, batch_representations = self.forward(data)

        # Compute contrastive loss
        loss, tqdm_dict = self.super_gmc_loss(output, target_data, batch_representations, temperature, batch_size, ep)
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
                 m2mix_type='pos',
                 beta_param=1.0,
                 single_mix=0.0,
                 all_mix=0.0,
                 multi_mix=0.0,
                 mix_schedule=0,
                 tot_ep=40
                 ):
            #         name=model_config["model"],
            # common_dim=model_config["common_dim"],
            # latent_dim=model_config["latent_dim"],
            # loss_type=model_config["loss_type"],
            # scenario=scenario,
            # #! mmixup-related
            # m2mix_type=model_config["m2mix_type"],
            # beta_param=model_config["beta_param"],
            # single_mix=model_config["single_mix"],
            # all_mix=model_config["all_mix"],
            # multi_mix=model_config["multi_mix"],
            # mix_schedule=model_config["mix_schedule"],
            # tot_ep=tot_ep,
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

        # Mixup-related parameter
        self.m2mix_type = m2mix_type
        self.beta_param = beta_param

        self.single_mix = single_mix
        self.all_mix = all_mix
        self.multi_mix = multi_mix
        self.mix_schedule = mix_schedule
        self.tot_ep = tot_ep

        # Classifier
        self.proj1 = nn.Linear(latent_dim, latent_dim)
        self.proj2 = nn.Linear(latent_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, 1)