# An object for implementing a task in DSG.
class DSGTask():
    # DSGTask() constructor
    def __init__(self, task_name, dataset=None, converge_lim=10, converge_threshold = 0.001, div_threshold=0.01):
        super(DSGTask, self).__init__()
        
        # String of task name (ex: "IMG")
        self.name = task_name
        # Best validation loss
        self.best_val_loss = 9e99
        # Epoch num for best validation loss
        self.best_loss_epoch = 0
        # Dataset for task
        self.dataset = dataset
        # Boolean to see if converged
        self.converged = False
        # Num rounds of no improvements to consider convergence
        self.convergence_limit = converge_lim
        # Boolean to determine if function is diverged
        self.diverged = False
        # Increase in val loss to consider as diverged
        self.divergence_threshold = div_threshold
        # Determine final convergence rounds
        self.final_round = False
        # Maintain a history of losses
        self.past_val_loss = []
        # Convergence rate
        self.convergence_rate = 1.0
        # Convergence Threshold
        self.convergence_threshold = converge_threshold

    # Helper to see if task is converged
    def is_converged(self):
        return self.converged
    
    # Helper to see if task is diverged
    def is_diverged(self):
        return self.diverged

    # Helper to set final rounds of convergence on min_lr
    def set_final_round(self):
        self.final_round = True

    def should_keep_training(self):
        return self.converged == False or self.diverged == True or self.set_final_round == True

    # Update function to perform updates on all tasks
    def update(self, cur_epoch, val_loss):
        self.past_val_loss.append(val_loss)
        if len(self.past_val_loss) < self.convergence_limit + 1:
            return
        self.convergence_rate = (self.past_val_loss[-self.convergence_limit - 1] - self.past_val_loss[-1])/ (self.convergence_limit)

        # Do nothing if on final rounds
        if self.final_round:
            return None

        if self.convergence_rate < self.convergence_threshold and self.convergence_rate > 0.0:
            self.diverged = False
            self.converged = True

        """ OLD VER
        # Update best val loss if val loss is better
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_loss_epoch = cur_epoch
            # Not diverged if returned back to previous best
            self.diverged = False

        if cur_epoch - self.best_loss_epoch + 1 > self.convergence_limit and not self.diverged:
            self.converged = True
        """

        # Check for divergence
        if self.converged and val_loss >= (self.best_val_loss + self.divergence_threshold):
            self.converged = False
            self.diverged = True

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
