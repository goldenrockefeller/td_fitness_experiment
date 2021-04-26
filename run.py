from td_fitness import *
import itertools

experiment_name = "A2"

# mods_to_mix = [
#     (short, medium, long),
#     (atc, asc, mtc, msc, imtc, imsc, qtc, qsc, uqtc, uqsc),
#     (no_noise, little_noise, some_noise, much_noise)
# ]

def soft_imtc(args):
    args["critic"] = InexactMidTrajCritic()
    args["critic"].learning_rate_scheme.learning_rate /= 22.360679775

def soft_imsc(args):
    args["critic"] = InexactMidSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedLearningRateScheme(args["critic"].core)
    args["critic"].learning_rate_scheme.time_horizon *= 22.360679775

mods_to_mix = [
    (long,),
    (biqtc, biqsc),
    (mega_noise,)
]


runners = [
    Runner(experiment_name, setup_combo)
    for setup_combo in itertools.product(*mods_to_mix)
]

for i in range(3):
    for runner in runners:
        runner.new_run()