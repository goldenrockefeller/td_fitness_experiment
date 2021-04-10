from td_fitness import *
import itertools

experiment_name = "Notest"

mods_to_mix = [
    (short, medium, long),
    (atc, asc, mtc, msc, imtc, imsc, qtc, qsc, uqtc, uqsc),
    (no_noise, little_noise, some_noise, much_noise)
]

mods_to_mix = [
    (medium,),
    (msc, imtc, imsc, qtc, qsc, uqtc, uqsc),
    (no_noise, little_noise, some_noise, much_noise)
]


runners = [
    Runner(experiment_name, setup_combo)
    for setup_combo in itertools.product(*mods_to_mix)
]


for runner in runners:
    runner.new_run()