import argparse

def get_config(case):
    if case == 'TRAIN_EEG-TEXT-BART': 
        # args config for training EEG-Text decoding
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-Text-Bart decoder')
        
        parser.add_argument('-m', '--model_name', help='Specify name of model', required=True)
        parser.add_argument('-LRinit', '--initial_learning_rate', help='Specify initial learning rate', type = float, required=True, default=5e-5)
        parser.add_argument('-LRmin', '--minimum_learning_rate', help='Specify minimum learning rate', required=True, type = float, default=5e-7)
        parser.add_argument('-LRgamma', '--gamma_learning_rate', help='Specify learning rate decay', required=False, type=float, default=2)
        parser.add_argument('-LRalt', '--alt_learning_rate', help='Specify if use learning rate alternator', type=bool, required=False, default=True)
        parser.add_argument('-device', '--device', help='Specify device', default="cuda")
        parser.add_argument('-ids', '--device_ids', help='Specify device ids', default=None)
        parser.add_argument('-log', '--log_dir', help='Specify tensorboard log directory')
        parser.add_argument('-ckpt', '--ckpt_dir', help='Specify checkpoint directory')
        parser.add_argument('-bsz', '--batch_size', help='Specify batch size', type = int)
        args = vars(parser.parse_args())
        if args["device_ids"] == None:
            args["device_ids"] = [0,1,2,3]
        else:
            args["device_ids"] = list(map(int, args["device_ids"].split(',')))

    return args