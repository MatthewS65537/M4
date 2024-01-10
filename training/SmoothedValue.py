class SmoothedValue():
    def __init__(self, smoothing_factor=0.9):
        """
        Initializes a SmoothedValue object with a given smoothing factor.

        Parameters:
        - smoothing_factor (float): The smoothing factor to be used for updating the value. Default is 0.9.
        """
        super(SmoothedValue, self).__init__()
        self.smoothing_factor = smoothing_factor
        self.value = None

    def update(self, new_value):
        """
        Updates the value of the SmoothedValue object using the given new value.

        Parameters:
        - new_value (float): The new value to be used for updating the value.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.smoothing_factor * self.value + (1 - self.smoothing_factor) * new_value

    def get(self):
        """
        Returns the current value of the SmoothedValue object.

        Returns:
        - float: The current value of the SmoothedValue object.
        """
        return self.value