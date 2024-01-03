import numpy as np

# An object for implementing a task in DSG.
class DSGTask():
    def __init__(self, task_name, dataset_tag, optimizer, learning_rate, criterion, converge_lim=10, converge_threshold=0.001, div_threshold=0.01):
        super(DSGTask, self).__init__()
        # Task Specs
        self.name = task_name
        self.dataset_tag = dataset_tag
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.criterion = criterion
        # DSG Specs
        self.best_val_loss = 9e99
        self.best_loss_epoch = 0
        self.converged = False
        self.convergence_limit = converge_lim
        self.diverged = False
        self.divergence_threshold = div_threshold
        self.final_round = False
        self.past_val_loss = []
        self.convergence_rate = 1.0
        self.convergence_threshold = converge_threshold
        
    def is_converged(self):
        return self.converged
    
    def is_diverged(self):
        return self.diverged

    def set_final_round(self):
        self.final_round = True

    def should_keep_training(self):
        return self.converged == False or self.diverged == True or self.final_round == True

    def reset_convergence(self):
        self.converged = False
        self.diverged = False
        self.past_val_loss = []
        self.convergence_rate = 1.0

    def reset_task(self):
        self.reset_convergence()
        self.best_val_loss = 9e99
        self.best_loss_epoch = 0
        self.final_round = False
        self.past_val_loss = []
        self.convergence_rate = 1.0

    def set_convergence_threshold(self, converge_threshold):
        self.convergence_threshold = converge_threshold

    def update(self, cur_epoch, val_loss):
        self.past_val_loss.append(val_loss)

        if len(self.past_val_loss) < self.convergence_limit + 1:
            return
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_loss_epoch = cur_epoch

        if val_loss > (self.best_val_loss * (1 + self.divergence_threshold)):
            self.diverged = True

        self.convergence_rate = (self.past_val_loss[-self.convergence_limit - 1] - self.past_val_loss[-1]) / (self.past_val_loss[-self.convergence_limit - 1] * self.convergence_limit)

        if self.final_round:
            return None

        if (self.convergence_rate < self.convergence_threshold) and (self.convergence_rate > 0.0) and (val_loss <= self.best_val_loss or (len(self.past_val_loss) > self.convergence_limit * 5 and val_loss <= self.best_val_loss + self.divergence_threshold)):
            self.diverged = False
            self.converged = True

# An object for interacting with numerous `DSGTask()` objects
class DSGTasks():
    def __init__(self):
        super(DSGTasks, self).__init__()
        self.tasks = []
        self.idx = 0
        self.num_tasks = 0

    # Adds a task to the list
    def add_task(self, task):
        self.tasks.append(task)
        self.num_tasks = len(self.tasks)

    # Resets idx pointer
    def reset_idx(self):
        self.idx = 0

    # Increments the idx pointer and returns its value (prior to incrementing)
    def next(self):
        self.idx += 1
        return self.idx - 1

    # Returns the number of tasks
    def length(self):
        return self.num_tasks

    # Returns a boolean of whether or not all tasks are converged
    def all_converged(self):
        for task in self.tasks:
            if not task.is_converged():
                return False
        return True

    # Returns a boolean of whether or not training should continue (returns True as long as one of them should keep training)
    def should_keep_training(self):
        for task in self.tasks:
            if task.should_keep_training():
                return True
        return False

    # Set convergence threshold for all tasks
    def set_convergence_threshold(self, converge_threshold):
        for task in self.tasks:
            task.set_convergence_threshold(converge_threshold)

    # Reset convergence for all tasks
    def reset_convergence(self):
        for task in self.tasks:
            task.reset_convergence()

    # Reset all tasks
    def reset_task(self):
        for task in self.tasks:
            task.reset_task()