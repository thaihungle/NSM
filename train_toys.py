import json
from tqdm import tqdm
import numpy as np

import os
import torch
from torch import nn, optim
from tensorboard_logger import configure, log_value
import torch.nn.functional as F

from nutm.nutm_warper import EncapsulatedNUTM
from datasets import CopyDataset, RepeatCopyDataset, MixCopyRepeatCopyDataset
from args import get_parser


args = get_parser().parse_args()



# ----------------------------------------------------------------------------
# -- initialize datasets, model, criterion and optimizer
# ----------------------------------------------------------------------------


task_params = json.load(open(args.task_json))
args.task_name = task_params['task']
if 'iter' in task_params:
    args.num_iters = task_params['iter']
log_dir = os.path.join(args.log_dir,args.task_name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
log_dir = os.path.join(log_dir, args.model_name+str(task_params['program_size']))
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

save_dir = os.path.join(args.save_dir,args.task_name+args.model_name+str(task_params['program_size']))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

save_dir = os.path.join(save_dir,"{}.pt".format(args.model_name))

configure(log_dir)

if "repeatcopy" in args.task_name:
    dataset = RepeatCopyDataset(task_params)
elif "copy" in args.task_name:
    dataset = CopyDataset(task_params)
elif "mix_cp_repeatcp" in args.task_name:
    dataset = MixCopyRepeatCopyDataset(task_params)


in_dim = dataset.in_dim
out_dim = dataset.out_dim




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

if torch.cuda.is_available():
    ntm.cuda()

print("====num params=====")

print(ntm.calculate_num_params())

print("========")

criterion = nn.BCELoss()
# As the learning rate is task specific, the argument can be moved to json file
optimizer = optim.RMSprop(ntm.parameters(),
                          lr=args.lr,
                          alpha=args.alpha,
                          momentum=args.momentum)


cur_dir = os.getcwd()


# ----------------------------------------------------------------------------
# -- basic training loop
# ----------------------------------------------------------------------------
losses = []
errors = []
loss_pls = []

best_loss = 10000

for iter in tqdm(range(args.num_iters)):
    optimizer.zero_grad()
    ntm.init_sequence(batch_size=args.batch_size)


    random_length = np.random.randint(task_params['min_seq_len'],
                                      task_params['max_seq_len']+1)

    data = dataset.get_sample_wlen(random_length, bs=args.batch_size)

    if torch.cuda.is_available():
        input, target = data['input'].cuda(), data['target'].cuda()
        out = torch.zeros(target.size()).cuda()
    else:
        input, target = data['input'], data['target']
        out = torch.zeros(target.size())


    for i in range(input.size()[0]):
        # to maintain consistency in dimensions as torch.cat was throwing error
        in_data = input[i]
        ntm(in_data)

    # passing zero vector as input while generating target sequence
    if torch.cuda.is_available():
        in_data = torch.zeros(input.size()).cuda()
    else:
        in_data = torch.zeros(input.size())

    for i in range(target.size()[0]):
        sout, _ = ntm(in_data[0])
        out[i] = F.sigmoid(sout)

    loss = criterion(out, target)

    args.pl1=args.pl1*args.decay_pl1
    args.pl2=args.pl2*args.decay_pl2

    loss2 = loss

    if task_params['program_size']>0 and args.pl1>0:
        loss_pl1 = ntm.program_loss_pl1()
        loss_pls.append(loss_pl1.item())
        loss2 = loss2 + args.pl1*loss_pl1

    if task_params['program_size']>0 and args.pl2>0:
        loss_pl2 = ntm.program_loss_pl2()
        loss_pls.append(loss_pl2.item())
        loss2 = loss2 + args.pl2 * loss_pl2

    losses.append(loss.item())
    loss2.backward()
    # clips gradient in the range [-10,10]. Again there is a slight but
    # insignificant deviation from the paper where they are clipped to (-10,10)
    if args.clip_grad>0:
        nn.utils.clip_grad_value_(ntm.parameters(), args.clip_grad)

    optimizer.step()

    binary_output = out.clone()
    if torch.cuda.is_available():
        binary_output = binary_output.detach().cpu().apply_(lambda x: 0 if x < 0.5 else 1).cuda()
    else:
        binary_output = binary_output.detach().apply_(lambda x: 0 if x < 0.5 else 1)

    error = torch.sum(torch.abs(binary_output - target)) / args.batch_size

    errors.append(error.item())

    # print(s[0])
    # ---logging---
    if iter % args.freq_val == 0:
        # adjust_learning_rate(optimizer, args.lr, iter, args.freq_val*4, 0.9)
        print('Iteration: %d\tLoss: %.2f\tError in bits per sequence: %.2f' %
              (iter, np.mean(losses), np.mean(errors)))
        mloss = np.mean(losses)

        if mloss<best_loss:
            # ---saving the model---
            torch.save(ntm.state_dict(), save_dir)
            best_loss = mloss
        log_value('train_loss', mloss, iter)
        log_value('bit_error_per_sequence', np.mean(errors), iter)
        if task_params['program_size']>0 and args.pl1 > 0:
            log_value('pl1', np.mean(loss_pls), iter)
        if task_params['program_size']>0 and args.pl2 > 0:
            log_value('pl2', np.mean(loss_pls), iter)
        losses = []
        errors = []
        loss_pls = []


