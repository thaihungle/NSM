import torch
from torch.utils.data import Dataset
import random
from datasets import CopyDataset, RepeatCopyDataset

class MixCopyRepeatCopyDataset(Dataset):
    """A Dataset class to generate random examples for the copy task. Each
    sequence has a random length between `min_seq_len` and `max_seq_len`.
    Each vector in the sequence has a fixed length of `seq_width`. The vectors
    are bounded by start and end delimiter flags.

    To account for the delimiter flags, the input sequence length as well
    width is two more than the target sequence.
    """

    def __init__(self, task_params):
        """Initialize a dataset instance for copy task.

        Arguments
        ---------
        task_params : dict
            A dict containing parameters relevant to copy task.
        """
        self.seq_width = task_params['seq_width']
        self.copy_dataset = CopyDataset(task_params)
        self.repeatcp_dataset = RepeatCopyDataset(task_params)
        self.in_dim = task_params['seq_width'] + 2
        self.out_dim = task_params['seq_width'] + 1


    def __len__(self):
        # sequences are generated randomly so this does not matter
        # set a sufficiently large size for data loader to sample mini-batches
        return 65536

    def __getitem__(self, idx):
        # idx only acts as a counter while generating batches.
        copy_data = self.copy_dataset[0]
        copy_input = copy_data['input']
        copy_target = copy_data['target']
        copy_target2 = torch.zeros([copy_target.size(0), copy_target.size(1)+1])
        copy_target2[:,:copy_target.size(1)] = copy_target
        copy_target = copy_target2
        arecall_data = self.repeatcp_dataset[0]
        arecall_input = arecall_data['input']
        arecall_target = arecall_data['target']

        task_order = torch.zeros([1, self.seq_width+2])

        if random.uniform(0, 1)>0.5:
            task_order[0,0]=1
            input_seq = torch.cat([task_order, copy_input, arecall_input], 0)
            target_seq = torch.cat([copy_target, arecall_target], 0)
        else:
            task_order[0,-1] = 1
            input_seq = torch.cat([task_order, arecall_input, copy_input], 0)
            target_seq = torch.cat([arecall_target, copy_target], 0)


        return {'input': input_seq, 'target': target_seq}

    def get_sample_wlen(self, seq_len, bs=1):
        copy_data = self.copy_dataset.get_sample_wlen(seq_len, bs)
        copy_input = copy_data['input']
        copy_target = copy_data['target']
        copy_target2 = torch.zeros([copy_target.size(0), copy_target.size(1), copy_target.size(2) + 1])
        copy_target2[:,:,:copy_target.size(2)] = copy_target
        copy_target = copy_target2
        arecall_data = self.repeatcp_dataset.get_sample_wlen(seq_len, bs)
        arecall_input = arecall_data['input']
        arecall_target = arecall_data['target']

        task_order = torch.zeros([1, bs, self.seq_width+2])

        if random.uniform(0, 1) > 0.5:
            task_order[:,:,0] = 1
            input_seq = torch.cat([task_order, copy_input, arecall_input], 0)
            target_seq = torch.cat([copy_target, arecall_target], 0)
        else:
            task_order[:,:,-1] = 1
            input_seq = torch.cat([task_order, arecall_input, copy_input], 0)
            target_seq = torch.cat([arecall_target, copy_target], 0)

        return {'input': input_seq, 'target': target_seq}

