class SmoothedValue():
    def __init__(self, smoothing_factor=0.9):
        super(SmoothedValue, self).__init__()
        self.smoothing_factor = smoothing_factor
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.smoothing_factor * self.value + (1 - self.smoothing_factor) * new_value

    def get(self):
        return self.value
