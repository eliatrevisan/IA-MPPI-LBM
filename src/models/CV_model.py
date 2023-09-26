import numpy as np

class ConstantVelocity():

    def __init__(self, args):
        self.args = args

    def predict(self, input_list):
        # Initialize array to save constant velocity predictions
        all_pred = []        
        for i in range(0, len(input_list)):
            input = input_list[i]
            pred_list = np.zeros([input.shape[0]], dtype=np.ndarray)
            for step in range(input.shape[0]):
                pred = self.cv(input[step,0], input[step,1], input[step,2], input[step,3]) 
                pred_list[step] = pred
            all_pred.append(pred_list)
        return np.array(all_pred)

    def cv(self, posx, posy, vx, vy):
        pred = []

        for i in range(self.args.prediction_horizon):
            posx = posx + vx*self.args.dt
            posy = posy + vy*self.args.dt
            pred.append([posx, posy])
        return np.array(pred)



