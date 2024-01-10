import torch
import torch.nn
import torch.nn.functional

class AutoTune():
    def __init__():
        self.parameter_names = []
        self.parameter_values = []
        self.parameter_min_diff = [] # -1 = doesn't matter
        self.parameter_gradients = []
        self.np_mode = False

    def add_parameter(name, val, min_diff):
        self.parameter_names.append(name)
        self.parameter_names.append(val)
        self.parameter_names.append(min_diff)
        self.parameter_gradients.append(0.0)
    
    def finalize():
        self.parameter_names = np.array(self.parameter_names)
        self.parameter_values = np.array(self.parameter_values)
        self.parameter_min_diff = np.array(self.parameter_min_diff)
        self.parameter_gradients = np.array(self.parameter_gradients)
        self.np_mode = True
        
    def search(model, args_dict, steps):
        if not np_mode:
            print("[ERROR] AutoTune not in Numpy mode. Cannot proceed.")
            assert(False)
        cur_step = 0

        while cur_step < steps:
            