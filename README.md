# NSM
source code for Neural Stored-program Memory  
arXiv version: https://arxiv.org/abs/1906.08862  
ICLR version: https://openreview.net/forum?id=rkxxA24FDr   
code reference for NTM tasks: https://github.com/vlgiitr/ntm-pytorch  
code reference for babi: https://github.com/JoergFranke/ADNC

# Single tasks
run command examples for training long Copy
``` 
NTM baseline: python train_toys.py -task_json=./tasks/copy_long_ntm.json  
NUTM: python train_toys.py -task_json=./tasks/copy_long.json  
```
for training Repeat Copy  
```
NTM baseline: python train_toys.py -task_json=./tasks/repeatcopy_ntm.json  
NUTM: python train_toys.py -task_json=./tasks/repeatcopy.json
```
for testing long Copy
``` 
NTM baseline: python evaluate_toys.py -task_json=./tasks/copy_long_ntm.json  
NUTM: python evaluate_toys.py -task_json=./tasks/copy_long.json  
```
# Sequencing tasks
training  
``` 
NTM baseline: python train_toys.py -task_json=./tasks/mix_cp_repeatcp_ntm.json -batch_size=16 
NUTM: python train_toys.py -task_json=./tasks/mix_cp_repeatcp.json -batch_size=16 
```

# Babi task (branch babi)
training
```
git checkout babi
cd scripts
python start_training.py
```