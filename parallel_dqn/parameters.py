import threading

import torch

from utils.atomic_int import AtomicInteger


# TODO: Fix sharding
class Parameters():
    lock = threading.Lock()
    parameters = None
    N = AtomicInteger(0)

    def __init__(self, parameters):
        with self.__class__.lock:
            if not self.__class__.parameters:
                self.__class__.parameters = parameters

    @classmethod
    def get_parameters(cls):
        return cls.parameters()

    @classmethod
    def send_parameters(cls, parameters, tau):
        with cls.lock:
            for target_param, local_param in zip(cls.parameters(), parameters):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)





