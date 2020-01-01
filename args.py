import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_name', default='ntm',
                        help='the name of the model')
    parser.add_argument('-task_json', type=str, default='tasks/copy_long.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('-att_mode', default='kv',
                        help='attention mode default is key value')
    parser.add_argument('-log_dir', default='./logs/',
                        help='path to log metrics')
    parser.add_argument('-save_dir', default='./saved_models/',
                        help='path to file with final model parameters')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='batch size of input sequence during training')
    parser.add_argument('-clip_grad', type=int, default=10,
                        help='clip gradient')
    parser.add_argument('-pl1', type=float, default=0.1,
                        help='decay rate for pl1 loss')
    parser.add_argument('-pl2', type=float, default=0,
                        help='decay rate for pl1 loss')
    parser.add_argument('-decay_pl1', type=float, default=0.9,
                        help='decay rate for pl1 loss')
    parser.add_argument('-decay_pl2', type=float, default=0,
                        help='decay rate for pl2 loss')
    parser.add_argument('-num_iters', type=int, default=100000,
                        help='number of iterations for training')
    parser.add_argument('-freq_val', type=int, default=200,
                        help='validation frequence')
    parser.add_argument('-num_eval', type=int, default=1000,
                        help='number of evaluation')
    parser.add_argument('-check_task', type=int, default=-1,
                        help='load checkpoint from task (only for continual learning)')
    parser.add_argument('-mode', type=str, default="train",
                        help='train or test (only for continual learning)')

    # todo: only rmsprop optimizer supported yet, support adam too
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('-alpha', type=float, default=0.95,
                        help='alpha for rmsprop optimizer')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help='beta1 constant for adam optimizer')
    parser.add_argument('-beta2', type=float, default=0.999,
                        help='beta2 constant for adam optimizer')
    return parser
