from a3c.a3c import A3C
from dqn.dqn import DQN
from dynaq.dynaq import DynaQ


class RLAlgorithmFactory:
    @staticmethod
    def get(name, *args):
        if name == "dqn":
            return DQN(*args)
        elif name == "a3c":
            return A3C(*args)
        elif name == "dynaq":
            return DynaQ(*args)
        else:
            raise Exception(name + " is not a valid agent")