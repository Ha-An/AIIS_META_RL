from AIIS_META.Sampler.base import SampleProcessor
from AIIS_META.Utils import utils
import numpy as np
import torch
class MetaSampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert isinstance(paths_meta_batch, dict), 'paths must be a dict'
        samples_data_meta_batch = []
        all_paths = []
        for meta_task, paths in paths_meta_batch.items():
            # fits baseline, compute advantages and stack path data
            samples_data, paths = self._compute_samples_data(paths)

            samples_data_meta_batch.append(samples_data)
            all_paths.extend(paths)
        

        '''E-MAML, 나중에 개발
        temp_overall_mean = []
        temp_overall_mstd = 
        for samples_data in samples_data_meta_batch:
            for rewards in samples_data['rewards']:
                    temp_overall_lst.append(torch.mean(torch.stack(rewards)))

        temp_overall_lst = torch.stack(temp_overall_lst)
        # 7) compute normalized trajectory-batch rewards (for E-MAML)
        overall_avg_reward = torch.mean(temp_overall_lst)
        overall_avg_reward_std = torch.std(temp_overall_lst)
        print(overall_avg_reward, overall_avg_reward_std)
        for samples_data in samples_data_meta_batch:
            samples_data['adj_avg_rewards'] = (samples_data['rewards'] - overall_avg_reward) / (overall_avg_reward_std + 1e-8)

        # 8) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix=log_prefix)
        '''
   
        return samples_data_meta_batch
    
    