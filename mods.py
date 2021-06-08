from td_fitness import *


def none(args):
    args["critic"] = MidTrajCritic()
    args["n_steps"] = 50
    args["domain_noise"] = 0.

def short(args):
    args["n_steps"] = 50

def medium(args):
    args["n_steps"] = 100

def long(args):
    args["n_steps"] = 500

def no_noise(args):
    args["domain_noise"] = 0.

def little_noise(args):
    args["domain_noise"] = 1./16

def some_noise(args):
    args["domain_noise"] = 1./4

def much_noise(args):
    args["domain_noise"] = 10.

def mega_noise(args):
    args["domain_noise"] = 100.

def set_horizon(horizon):
    def inner(args):
        args["horizon"] = horizon

    return inner


horizon_4 = set_horizon(4)
horizon_8 = set_horizon(8)
horizon_16 = set_horizon(16)
horizon_32 = set_horizon(32)



def mtc(args):
    args["critic"] = MidTrajCritic()
    args["critic"].learning_rate_scheme = TrajMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def msc(args):
    args["critic"] = MidSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedMonteLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imtc(args):
    args["critic"] = InexactMidTrajCritic()
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def imsc(args):
    args["critic"] = InexactMidSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qtc(args):
    args["critic"] = QTrajCritic()
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def qsc(args):
    args["critic"] = QSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)

    args["critic"].time_horizon = args["horizon"]
def biqtc(args):
    args["critic"] = BiQTrajCritic()
    args["critic"].learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def biqsc(args):
    args["critic"] = BiQSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].core)
    args["critic"].time_horizon = args["horizon"]

def uqtc(args):
    args["critic"] = UqTrajCritic()
    args["critic"].u_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def uqsc(args):
    args["critic"] = UqSteppedCritic(args["n_steps"])
    args["critic"].u_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].u_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].u_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def atc(args):
    args["critic"] = ATrajCritic()
    args["critic"].v_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = TrajTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]

def asc(args):
    args["critic"] = ASteppedCritic(args["n_steps"])
    args["critic"].v_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].v_critic.core, True)
    args["critic"].q_critic.learning_rate_scheme = SteppedTabularLearningRateScheme(args["critic"].q_critic.core)
    args["critic"].v_critic.time_horizon = args["horizon"]
    args["critic"].q_critic.time_horizon = args["horizon"]