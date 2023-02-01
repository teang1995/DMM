from tqdm import tqdm
from pytorch_lightning import LightningModule
from gmc_code.supervised_dmm.modules.trainers.model_evaluation_metrics import *
import pdb


PERFORMANCE_GOAL = {'mae_T':0.7081,'corr_T':0.5814,'fscore_T':0.7800,'acc_T':0.7801,
                    'mae_A':0.8296,'corr_A':0.2741,'fscore_A':0.6787,'acc_A':0.6635,
                    'mae_V':0.8427,'corr_V':0.2634,'fscore_V':0.6586,'acc_V':0.6536,
                    'mae_TA':0.7086,'corr_TA':0.5625,'fscore_TA':0.7745,'acc_TA':0.7702,
                    'mae_TV':0.7013,'corr_TV':0.5556,'fscore_TV':0.7720,'acc_TV':0.7693,
                    'mae_AV':0.8039,'corr_AV':0.3140,'fscore_AV':0.6841,'acc_AV':0.6718,
                    'mae_TAV':0.618,'corr_TAV':0.6705,'fscore_TAV':0.8068,'acc_TAV':0.8066}

class ModelEvaluation(LightningModule):
    def __init__(self, model_name, model, scenario, test_loader, sacred_logger, modalities=None, seed=0):
        super(ModelEvaluation, self).__init__()

        self.scenario = scenario
        self.model_name = model_name

        self.model = model
        self.model.eval()

        self.test_modalities = modalities
        self.test_loader = test_loader
        self.sacred_logger = sacred_logger
        self.seed = seed

    def evaluate(self):    
        #pdb.set_trace()
        if self.test_modalities[0] < 0:
            entire_mae = []; mae_win_count = 0
            entire_corr = []; corr_win_count = 0
            entire_fscore = []; fscore_win_count = 0
            entire_acc = []; acc_win_count = 0
            test_all_scenarios = [[0,1,2], [0], [1], [2], [0,1], [0,2], [1,2]]
            for scen in test_all_scenarios:
                if scen == [0]: mod_n = 'T'
                if scen == [1]: mod_n = 'A'
                if scen == [2]: mod_n = 'V'
                if scen == [0,1]: mod_n = 'TA'
                if scen == [0,2]: mod_n = 'TV'
                if scen == [1,2]: mod_n = 'AV'
                if scen == [0,1,2]: mod_n = 'TAV'
                results = []
                truths = []
                with torch.no_grad():
                    for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.test_loader)):

                        sample_ind, text, audio, vision = batch_X
                        data = [text, audio, vision]
                        target_data = batch_Y.squeeze(-1)  # if num of labels is 1

                        # Drop modalities (if required)
                        input_data = []
                        for j in range(len(data)):
                            if j not in scen:
                                input_data.append(None)
                            else:
                                input_data.append(data[j])

                        # Parallel model
                        preds = self.model.encode(input_data)

                        if self.scenario == 'iemocap':
                            preds = preds.view(-1, 2)
                            target_data = target_data.view(-1)

                        # Collect the results into dictionary
                        results.append(preds)
                        truths.append(target_data)

                    results = torch.cat(results)
                    truths = torch.cat(truths)

                    if self.scenario == "mosei":
                        m, c, f, a = eval_mosei(results, truths, self.sacred_logger, True, self.model_name, scen, self.seed)
                        entire_mae.append(m)
                        if m > PERFORMANCE_GOAL[f'mae_{mod_n}']: mae_win_count += 1
                        entire_corr.append(c)
                        if c > PERFORMANCE_GOAL[f'corr_{mod_n}']: corr_win_count += 1
                        entire_fscore.append(f)
                        if f > PERFORMANCE_GOAL[f'fscore_{mod_n}']: fscore_win_count += 1
                        entire_acc.append(a)
                        if a > PERFORMANCE_GOAL[f'acc_{mod_n}']: acc_win_count += 1
                    elif self.scenario == 'mosi':
                        eval_mosi(results, truths, self.sacred_logger, True, self.model_name, scen, self.seed)
            tot_win = mae_win_count + corr_win_count + fscore_win_count + acc_win_count
        else:
            results = []
            truths = []
            with torch.no_grad():
                for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.test_loader)):

                    sample_ind, text, audio, vision = batch_X
                    data = [text, audio, vision]
                    target_data = batch_Y.squeeze(-1)  # if num of labels is 1

                    # Drop modalities (if required)
                    input_data = []
                    if self.test_modalities is not None:
                        for j in range(len(data)):
                            if j not in self.test_modalities:
                                input_data.append(None)
                            else:
                                input_data.append(data[j])
                    else:
                        input_data = data

                    # Parallel model
                    preds = self.model.encode(input_data)

                    if self.scenario == 'iemocap':
                        preds = preds.view(-1, 2)
                        target_data = target_data.view(-1)

                    # Collect the results into dictionary
                    results.append(preds)
                    truths.append(target_data)

                results = torch.cat(results)
                truths = torch.cat(truths)

                if self.scenario == "mosei":
                    eval_mosei(results, truths, self.sacred_logger, True, self.model_name, self.test_modalities, self.seed)
                elif self.scenario == 'mosi':
                    eval_mosi(results, truths, self.sacred_logger, True, self.model_name, self.test_modalities, self.seed)

    def get_emb(self, mod=[0,2]):    
        # if self.test_modalities[0] < 0:
        #     entire_mae = []; mae_win_count = 0
        #     entire_corr = []; corr_win_count = 0
        #     entire_fscore = []; fscore_win_count = 0
        #     entire_acc = []; acc_win_count = 0
        #     test_all_scenarios = [[0,1,2], [0], [1], [2], [0,1], [0,2], [1,2]]
        #     for scen in test_all_scenarios:
        #         if scen == [0]: mod_n = 'T'
        #         if scen == [1]: mod_n = 'A'
        #         if scen == [2]: mod_n = 'V'
        #         if scen == [0,1]: mod_n = 'TA'
        #         if scen == [0,2]: mod_n = 'TV'
        #         if scen == [1,2]: mod_n = 'AV'
        #         if scen == [0,1,2]: mod_n = 'TAV'
                
        results = []
        truths = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.test_loader)):
                sample_ind, text, audio, vision = batch_X
                data = [text, audio, vision]
                target_data = batch_Y.squeeze(-1)  # if num of labels is 1

                # Drop modalities (if required)
                input_data = []
                for j in range(len(data)):
                    if j not in mod:
                        input_data.append(None)
                    else:
                        input_data.append(data[j])

                # Get representations
                embs = self.model.encode(input_data, out_z=True)

                # Collect the results into dictionary
                results.append(embs)
                truths.append(target_data)

            results = torch.cat(results)
            truths = torch.cat(truths)

        return results, truths