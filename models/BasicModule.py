from abc import ABC
import torch
import time


class BasicModule(torch.nn.Module, ABC):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.name = self.__class__.__name__

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self):
        path = "checkpoints/" + self.name + "-"
        cur_time = time.strftime("%m_%d_%H:%M:%S") + ".pth"
        torch.save(self.state_dict(), path + cur_time)

    def save_with_label(self, label):
        path = "checkpoints/" + self.name + "-" + label + '-'
        cur_time = time.strftime("%m_%d_%H:%M:%S") + ".pth"
        torch.save(self.state_dict(), path + cur_time)
