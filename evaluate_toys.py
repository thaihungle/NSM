import json
import os

import torch
from torch import nn
from tqdm import tqdm

import torch.nn.functional as F

from nutm.nutm_warper import EncapsulatedNUTM
from datasets import CopyDataset, RepeatCopyDataset, MixCopyRepeatCopyDataset
from args import get_parser
import visual_tool as vis
import numpy as np
import random



args = get_parser().parse_args()

task_params = json.load(open(args.task_json))
args.task_name = task_params['task']

criterion = nn.BCELoss()


'''
# For NGram and Priority sort task parameters need not be changed.
'''

if "repeatcopy" in args.task_name:
    # ---Evaluation parameters for RepeatCopy task---
    # (Sequence length generalisation)
    task_params['min_seq_len'] = 10
    task_params['max_seq_len'] = 20
    # (Number of repetition generalisation)
    task_params['min_repeat'] = 10
    task_params['max_repeat'] = 20
    dataset = RepeatCopyDataset(task_params)
elif "copy_long" in args.task_name:
    # ---Evaluation parameters for Copy task---
    task_params['min_seq_len'] = 199
    task_params['max_seq_len'] = 200
    dataset = CopyDataset(task_params)
elif "mix_cp_repeatcp" in args.task_name:
    task_params['min_seq_len'] = 10
    task_params['max_seq_len'] = 20
    task_params['min_repeat'] = 10
    task_params['max_repeat'] = 15
    dataset = MixCopyRepeatCopyDataset(task_params)



in_dim = dataset.in_dim
out_dim = dataset.out_dim

save_dir = os.path.join(args.save_dir,args.task_name+args.model_name+str(task_params['program_size']))
save_dir = os.path.join(save_dir,"{}.pt".format(args.model_name))


cur_dir = os.getcwd()
PATH = os.path.join(cur_dir, args.save_dir)


"""
For the Copy task, input_size: seq_width + 2, output_size: seq_width
For the RepeatCopy task, input_size: seq_width + 2, output_size: seq_width + 1
For the Associative task, input_size: seq_width + 2, output_size: seq_width
For the NGram task, input_size: 1, output_size: 1
For the Priority Sort task, input_size: seq_width + 1, output_size: seq_width
"""

ntm = EncapsulatedNUTM(
    num_inputs=in_dim,
    num_outputs=out_dim,
    controller_size=task_params['controller_size'],
    controller_layers =1,
    num_heads = task_params['num_heads'],
    N=task_params['memory_units'],
    M=task_params['memory_unit_size'],
    program_size =task_params['program_size'],
    pkey_dim=task_params['pkey_dim'])
ntm.set_att_mode(args.att_mode)

torch.manual_seed(1111)
torch.cuda.manual_seed(1111)
np.random.seed(1111)
random.seed(1111)


if torch.cuda.is_available():
    ntm.cuda()

ntm.load_state_dict(torch.load(save_dir))
print("====num params=====")

print(ntm.calculate_num_params())
print("========")
# -----------------------------------------------------------------------------
# --- evaluation
# -----------------------------------------------------------------------------
losses = []
errors = []
opt_errors = []

min_err = 10000000

for num_eval in tqdm(range(args.num_eval)):
    ntm.init_sequence(batch_size=1)
    data = dataset[0]  # 0 is a dummy index
    if torch.cuda.is_available():
        input, target = data['input'].cuda(), data['target'].cuda()
        out = torch.zeros(target.size()).cuda()
    else:
        input, target = data['input'], data['target']
        out = torch.zeros(target.size())

    # -----------------------------------------------------------------------------
    # loop for other tasks
    # -----------------------------------------------------------------------------
    read_ws = []
    write_ws = []
    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = torch.unsqueeze(input[i], 0)

        _, states = ntm(in_data)

        if "ntm" in args.model_name:
            head_states = states[2]
            read_w = head_states[0]
            write_w = head_states[1]
            read_ws.append(read_w)
            write_ws.append(write_w)

    # print('encoding stop-decoding start')

    # passing zero vector as the input while generating target sequence
    if torch.cuda.is_available():
        in_data = torch.unsqueeze(torch.zeros(input.size()[1]).cuda(), 0)
    else:
        in_data = torch.unsqueeze(torch.zeros(input.size()[1]), 0)

    for i in range(target.size()[0]):
        sout, states = ntm(in_data)
        out[i] = F.sigmoid(sout)

        if "ntm" in args.model_name:
            head_states = states[2]
            read_w = head_states[0]
            write_w = head_states[1]
            read_ws.append(read_w)
            write_ws.append(write_w)

    loss = criterion(out, target)
    binary_output = out.clone()

    error = torch.sum(torch.abs(binary_output - target)) / args.batch_size

    # print(binary_output)
    # print(target)
    # sequence prediction error is calculted in bits per sequence

    if "ntm" in args.model_name:
        debug_mem = {"read_weights": torch.cat(read_ws, 0),
                     "write_weights": torch.cat(write_ws, 0)}

    losses.append(loss.item())
    errors.append(error.item())

    if 'ntm' in args.model_name:
        if error.item() < min_err:
            meta = ntm.get_read_meta_info()
            min_err = error.item()
            inlen = input.shape[0] - 0.5

            all_data = torch.cat([input[:, :binary_output.shape[1]],
                                  binary_output[:, :binary_output.shape[1]]], 0).detach().cpu()

# ---logging---
print('Loss: %.2f\tError in bits per sequence: %.2f' % (np.mean(losses), np.mean(errors)))
print('Best Loss: %.2f\tError in bits per sequence: %.2f' % (np.min(losses), np.min(errors)))
if opt_errors:
    print('Opt Error in bits per sequence: %.2f' % (np.mean(opt_errors)))

# vis.plot_state_space(meta["css"][0])

if task_params["program_size"] > 0:

    vis.plot_meta(meta["read_program_weights"][0],
                  meta["write_program_weights"][0],
                  meta["read_data_weights"][0],
                  meta["write_data_weights"][0], all_data,
                  inlen,
                  int(np.min(errors))
                  )
else:
    vis.plot_inout(meta["read_data_weights"][0],
                   meta["write_data_weights"][0], all_data,
                   inlen,
                   np.min(errors))
