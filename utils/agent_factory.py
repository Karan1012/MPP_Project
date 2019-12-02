from a3c.agent import A3CAgent
from dynaq.agent import DynaQAgent
from parallel_dqn.agent import ParallelDQNAgent


class AgentFactory:
    @staticmethod
    def get_agent(name, *args):
        if name == "dqn":
            return ParallelDQNAgent(*args)
        elif name == "a3c":
            return A3CAgent(*args)
        elif name == "dynaq":
            return DynaQAgent(*args)
        else:
            raise Exception(name + " is not a valid agent")