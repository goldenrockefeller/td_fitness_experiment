from td_fitness import *
import itertools

experiment_name = "A1"

# mods_to_mix = [
#     (short, medium, long),
#     (atc, asc, mtc, msc, imtc, imsc, qtc, qsc, uqtc, uqsc),
#     (no_noise, little_noise, some_noise, much_noise)
# ]

mods_to_mix = [
    (long,),
    (imtc, imsc, qtc, qsc, uqtc, uqsc,),
    (mega_noise,)
]


runners = [
    Runner(experiment_name, setup_combo)
    for setup_combo in itertools.product(*mods_to_mix)
]

for i in range(3):
    for runner in runners:
        runner.new_run()