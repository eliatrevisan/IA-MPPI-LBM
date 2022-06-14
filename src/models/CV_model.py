import numpy as np

class ConstantVelocity():

    def __init__(self, args):
        self.args = args

    def predict(self, input):
        """
        Input should be current position / velocity
        Output should be predicted trajectory according to constant velocity model
        """