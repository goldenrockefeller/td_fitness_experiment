
import numpy as np
from random import shuffle
import os
import errno
import datetime
import itertools
import glob
import datetime as dt
from shutil import copy
import csv
import time


# TODO
# 9 (18) Learning Algorithms
# - [DONE] Fitness Critic (gradient descent through aggregation)
# - [DONE] Fitness Critic (gradient descent through aggregation step)
# - [DONE] Fitness Critic (gradient descent through intermediate critic)
# - [DONE] Fitness Critic (gradient descent through intermediate critic step)
# - [DONE] Q Critic (Q table for whole trajectory)
# - [DONE] Q Critic (Q table per step)
# - [DONE] P + Q Critic (for whole trajectory)
# - [DONE] P + Q Critic (per step)
# Testing for
# - Learning speed and Policy Performance
# - - [DONE] What is max reward
# - - [DONE] Actual Performance Curve
# - - [DONE] Expected Performance Curve
# - - Critic Evaluation Curve
# - Critic Accuracy (see below)
# - - [Done] Compared to Actual Performance for critic
# - - [DONE] Compared to Expected Performance for critic given policy
# Domain parameters
# - [DONE] Increasing domain length
# - [DONE] Increasing reward noise
# [DONE] check manual expected performance
# [DONE] run generator (epochs = 100 * n_steps)
# [DONE] Acceleration of Algorithms
# Replace P critic with U Critic
# Use Nesterov
# cythonize code
# use TD-lambda learning
# cythonize code

def list_sum(my_list):
    val = 0.

    for d in my_list:
        val += d

    return val

def list_multiply(my_list, val):
    new_list = my_list.copy()

    for id, d in enumerate(my_list):
        my_list[id] = d * val

    return my_list

random_cache_size = 10000
random_uniform_counter = 0
np_random_uniform_cache = np.random.random(random_cache_size)
def random_uniform():
    global random_uniform_counter, np_random_uniform_cache, random_cache_size

    if random_uniform_counter >= random_cache_size:
        random_uniform_counter = 0
        np_random_uniform_cache = np.random.random(random_cache_size)

    val = np_random_uniform_cache[random_uniform_counter]
    random_uniform_counter += 1
    return val

random_normal_counter = 0
np_random_normal_cache = np.random.normal(size = (random_cache_size,))
def random_normal():
    global random_normal_counter, np_random_normal_cache, random_cache_size

    if random_normal_counter >= random_cache_size:
        random_normal_counter = 0
        np_random_normal_cache = np.random.normal(size = (random_cache_size,))

    val = np_random_normal_cache[random_normal_counter]
    random_normal_counter += 1
    return val


def mdp_policy_values(P_policy, P_transition, R_expected, n_steps):
    """
    P_policy [action, state] chance of choosing action when in state
    P_transition [action, state, next_state] chance of reaching next_state
      when choosing action in state
    R_expected [action, state] expected reward for choosing action in state
    """
    n_states = P_policy.shape[1]
    n_actions = P_policy.shape[0]
    V = np.zeros(n_states)


    P_transition_reduced = 0. * P_transition[0]
    R_expected_reduced = 0. * R_expected[0]

    for action_id in range(n_actions):
        P_transition_reduced += P_policy[action_id] * P_transition[action_id]
        R_expected_reduced += P_policy[action_id] * R_expected[action_id]


    for step_id in range(n_steps):
        prev_V = 1. * V
        V *= 0
        for action_id in range(n_actions):
            V += (
                P_policy[action_id]
                * (
                    np.matmul(P_transition[action_id], prev_V)
                    + R_expected[action_id]
                )
            )
        # V = np.matmul(P_transition_reduced, V) + R_expected_reduced

    return V

def mdp_optimal_values(P_transition, R_expected, n_steps):
    """
    P_policy [action, state] chance of choosing action when in state
    P_transition [action, state, next_state] chance of reaching next_state
      when choosing action in state
    R_expected [action, state] expected reward for choosing action in state
    """
    n_states = P_transition.shape[1]
    n_actions = P_transition.shape[0]
    V = np.zeros(n_states)
    prev_V = 1. * V
    # R_expected [action, state] expected reward for choosing action in state


    for step_id in range(n_steps):
        prev_V = 1. * V
        V = np.matmul(P_transition[0], prev_V) + R_expected[0]
        for action_id in range(n_actions):
            V = (
                np.maximum(
                    V,
                    (
                        np.matmul(P_transition[action_id], prev_V)
                        + R_expected[action_id]
                    )
                )
            )



    return V


class StateEnum:
    def __init__(self):
        self.UP = 0
        self.HOME = 1
        self.DOWN = 2

State =  StateEnum()

class ActionEnum:
    def __init__(self):
        self.A = 0
        self.B = 1

Action = ActionEnum()

class BasicLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate


    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, states, actions):
        return [self.learning_rate for _ in range(len(states))]

class ReducedLearningRateScheme():
    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def copy(self):
        scheme = self.__class__()
        scheme.learning_rate = self.learning_rate

        return scheme

    def learning_rates(self, states, actions):
        n_steps =  len(states)
        return [self.learning_rate / n_steps for _ in range(len(states))]

class TrajMonteLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 100.):
        self.denoms = {key: 0. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon

    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_updates_elapsed = 0
        scheme.time_horizon = self.time_horizon

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]
        visitation = {}
        local_pressure = {}

        sum_of_sqr_rel_pressure = 0.
        sum_rel_pressure = 0.


        for state, action in zip(states, actions):
            self.denoms[(state, action)] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(state, action)])
            )
            self.last_update_seen[(state, action)] = self.n_updates_elapsed

            visitation[(state, action)] = 0.
            local_pressure[(state, action)] = 0.


        for state, action in zip(states, actions):
            visitation[(state, action)] += 1.


        for key in visitation:
            local_pressure[key] = (
                visitation[key]
                * visitation[key]
            )

            relative_pressure = (
                local_pressure[key]
                / (
                    local_pressure[key]
                    +  self.denoms[key]
                )
            )

            sum_of_sqr_rel_pressure += relative_pressure * relative_pressure
            sum_rel_pressure += relative_pressure

        step_size = sum_rel_pressure * sum_rel_pressure/ sum_of_sqr_rel_pressure


        for key in visitation:
            self.denoms[key] += step_size * local_pressure[key]


        for step_id, (state, action) in enumerate(zip(states, actions)):
            rates[step_id] = 1. / self.denoms[(state, action)]

        self.n_updates_elapsed += 1
        return rates

class SteppedMonteLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_updates_elapsed = 0
        scheme.time_horizon = self.time_horizon

        return scheme

    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]
        visited = []

        sum_of_sqr_rel_pressure = 0.
        sum_rel_pressure = 0.


        for step_id, (state, action) in enumerate(zip(states, actions)):
            self.denoms[(state, action)][step_id] *= (
                (1. - 1. / self.time_horizon)
                ** (self.n_updates_elapsed - self.last_update_seen[(state, action)][step_id])
            )
            self.last_update_seen[(state, action)][step_id] = self.n_updates_elapsed

            visited.append(((state, action), step_id))

        for key, step_id in visited:
            relative_pressure = 1./ (1. +  self.denoms[key][step_id])

            sum_of_sqr_rel_pressure += relative_pressure * relative_pressure
            sum_rel_pressure += relative_pressure

        step_size = (
            sum_rel_pressure * sum_rel_pressure
            / sum_of_sqr_rel_pressure
        )

        for key, step_id in visited:
            self.denoms[key][step_id] += step_size


        for step_id, (state, action) in enumerate(zip(states, actions)):
            rates[step_id] = 1. / (self.denoms[(state, action)][step_id])

        self.n_updates_elapsed += 1
        return rates

class TrajTabularLearningRateScheme():
    def __init__(self, ref_model, has_only_state_as_key = False, time_horizon = 100.):
        self.denoms = {key: 0. for key in ref_model}
        self.last_update_seen = {key: 0 for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_state_as_key = has_only_state_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.last_update_seen = self.last_update_seen.copy()
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_state_as_key = self.has_only_state_as_key

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]

        for state, action in zip(states, actions):
            if self.has_only_state_as_key:
                self.denoms[state] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[state])
                )
                self.last_update_seen[state] = self.n_updates_elapsed

            else:
                self.denoms[(state, action)] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(state, action)])
                )
                self.last_update_seen[(state, action)] = self.n_updates_elapsed

        for state, action in zip(states, actions):
            if self.has_only_state_as_key:
                self.denoms[state] += 1

            else:
                self.denoms[(state, action)] += 1

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                rates[step_id] = 1. / self.denoms[state]

            else:
                rates[step_id] = 1. / self.denoms[(state, action)]

        self.n_updates_elapsed += 1
        return rates



class SteppedTabularLearningRateScheme():

    def __init__(self, ref_model, has_only_state_as_key = False, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.last_update_seen =  {key: [0 for _ in range(len(ref_model[key]))] for key in ref_model}
        self.n_updates_elapsed = 0
        self.time_horizon = time_horizon
        self.has_only_state_as_key = has_only_state_as_key


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.last_update_seen = {key : self.last_update_seen[key].copy() for key in self.last_update_seen}
        scheme.n_updates_elapsed = self.n_updates_elapsed
        scheme.time_horizon = self.time_horizon
        scheme.has_only_state_as_key = self.has_only_state_as_key

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                self.denoms[state][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[state][step_id])
                )
                self.last_update_seen[state][step_id] = self.n_updates_elapsed

            else:
                self.denoms[(state, action)][step_id] *= (
                    (1. - 1. / self.time_horizon)
                    ** (self.n_updates_elapsed - self.last_update_seen[(state, action)][step_id])
                )
                self.last_update_seen[(state, action)][step_id] = self.n_updates_elapsed

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                self.denoms[state][step_id] += 1

            else:
                self.denoms[(state, action)][step_id] += 1

        for step_id, (state, action) in enumerate(zip(states, actions)):
            if self.has_only_state_as_key:
                rates[step_id] = 1. / self.denoms[state][step_id]

            else:
                rates[step_id] = 1. / self.denoms[(state, action)][step_id]

        self.n_updates_elapsed += 1
        return rates



class TrajCritic():
    def __init__(self):
        self.learning_rate_scheme = ReducedLearningRateScheme()
        self.core = {}
        self.core[(State.UP, Action.A)] = 0.
        self.core[(State.HOME, Action.A)] = 0.
        self.core[(State.DOWN, Action.A)] = 0.
        self.core[(State.UP, Action.B)] = 0.
        self.core[(State.HOME, Action.B)] = 0.
        self.core[(State.DOWN, Action.B)] = 0.


    def copy(self):
        critic = self.__class__()
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = self.core.copy()

        return critic

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[(state, action)]
        return evals

class SteppedCritic():
    def __init__(self, n_steps):
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core = {}
        self.core[(State.UP, Action.A)] = [0. for _ in range(n_steps)]
        self.core[(State.HOME, Action.A)] = [0. for _ in range(n_steps)]
        self.core[(State.DOWN, Action.A)] = [0. for _ in range(n_steps)]
        self.core[(State.UP, Action.B)] = [0. for _ in range(n_steps)]
        self.core[(State.HOME, Action.B)] = [0. for _ in range(n_steps)]
        self.core[(State.DOWN, Action.B)] = [0. for _ in range(n_steps)]
        self.n_steps = n_steps

    def copy(self):
        critic = self.__class__(self.n_steps)
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[(state, action)]
            evals[step_id] = state_evals[step_id]
        return evals

class AveragedTrajCritic(TrajCritic):
    def eval(self, states, actions):
        return TrajCritic.eval(self, states, actions) / len(states)


class AveragedSteppedCritic(SteppedCritic):
    def eval(self, states, actions):
        return SteppedCritic.eval(self, states, actions) / len(states)


class MidTrajCritic(TrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        estimate = self.eval(states, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id]
            self.core[(state, action)] += delta

class MidSteppedCritic(SteppedCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        estimate = self.eval(states, actions)

        error = fitness - estimate

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            delta = error * learning_rates[step_id]
            self.core[(state, action)][step_id] += delta

class InexactMidTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.core[(state, action)] += delta


class InexactMidSteppedCritic(AveragedSteppedCritic):
    def update(self, states, actions, rewards):
        n_steps = len(states)

        fitness = list_sum(rewards)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        for step_id in range(n_steps):
            state = states[step_id]
            action = actions[step_id]
            estimate = step_evals[step_id]
            error = fitness - estimate
            delta = error * learning_rates[step_id]
            self.core[(state, action)][step_id] += delta

class QTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[(states[-1], actions[-1])] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[(states[step_id], actions[step_id])] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )


class QSteppedCritic(AveragedSteppedCritic):



    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[(states[-1], actions[-1])][-1] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[(states[step_id], actions[step_id])][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )

class BiQTrajCritic(AveragedTrajCritic):

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        if n_steps >= 2:
            self.core[(states[-1], actions[-1])] += (
                learning_rates[-1]
                * (
                    rewards[-1]
                    + step_evals[-2]
                    - step_evals[-1]
                )
            )

            self.core[(states[0], actions[0])] += (
                learning_rates[0]
                * (
                    rewards[0]
                    + step_evals[1]
                    - step_evals[0]
                )
            )


            for step_id in range(1, n_steps - 1):
                self.core[(states[step_id], actions[step_id])] += (
                    learning_rates[step_id]
                    * (
                        rewards[step_id]
                        + 0.5 * step_evals[step_id + 1]
                        + 0.5 * step_evals[step_id - 1]
                        - step_evals[step_id]
                    )
                )
        else:
            # nsteps = 1
            raise (
                NotImplementedError(
                    "BiQ is currently implemented for when the number of steps "
                    "is greater than 1."
                )
            )


class BiQSteppedCritic(AveragedSteppedCritic):



    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        if n_steps >= 2:
            self.core[(states[-1], actions[-1])][-1] += (
                learning_rates[-1]
                * (
                    rewards[-1]
                    + step_evals[-2]
                    - step_evals[-1]
                )
            )

            self.core[(states[0], actions[0])][0] += (
                learning_rates[0]
                * (
                    rewards[0]
                    + step_evals[1]
                    - step_evals[0]
                )
            )


            for step_id in range(1, n_steps - 1):
                self.core[(states[step_id], actions[step_id])][step_id] += (
                    learning_rates[step_id]
                    * (
                        rewards[step_id]
                        + 0.5 * step_evals[step_id + 1]
                        + 0.5 * step_evals[step_id - 1]
                        - step_evals[step_id]
                    )
                )
        else:
            # nsteps = 1
            raise (
                NotImplementedError(
                    "BiQ is currently implemented for when the number of steps "
                    "is greater than 1."
                )
            )

class VTrajCritic(AveragedTrajCritic):
    def __init__(self):
        self.core = {}
        self.core[State.UP] = 0.
        self.core[State.HOME] = 0.
        self.core[State.DOWN] = 0.
        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[state]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[-1]] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[states[step_id]] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )


class VSteppedCritic(AveragedSteppedCritic):

    def __init__(self, n_steps):
        self.core = {}
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core[State.UP] = [0. for _ in range(n_steps)]
        self.core[State.HOME] = [0. for _ in range(n_steps)]
        self.core[State.DOWN] = [0. for _ in range(n_steps)]

    def copy(self):
        critic = self.__class__(len(self.core[State.UP]))
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic


    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[state]
            evals[step_id] = state_evals[step_id]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[-1]][-1] += (
            learning_rates[-1]
            * (
                rewards[-1]
                - step_evals[-1]
            )
        )

        for step_id in range(n_steps - 1):
            self.core[states[step_id]][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id]
                    + step_evals[step_id + 1]
                    - step_evals[step_id]
                )
            )

class UTrajCritic(AveragedTrajCritic):
    def __init__(self):
        self.core = {}
        self.core[State.UP] = 0.
        self.core[State.HOME] = 0.
        self.core[State.DOWN] = 0.
        self.learning_rate_scheme = ReducedLearningRateScheme()

    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            evals[step_id] = self.core[state]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[0]] += (
            learning_rates[0]
            * (
                - step_evals[0]
            )
        )

        for step_id in range(1, n_steps):
            self.core[states[step_id]] += (
                learning_rates[step_id]
                * (
                    rewards[step_id - 1]
                    + step_evals[step_id - 1]
                    - step_evals[step_id]
                )
            )

class USteppedCritic(AveragedSteppedCritic):

    def __init__(self, n_steps):
        self.core = {}
        self.learning_rate_scheme = BasicLearningRateScheme()
        self.core[State.UP] = [0. for _ in range(n_steps)]
        self.core[State.HOME] = [0. for _ in range(n_steps)]
        self.core[State.DOWN] = [0. for _ in range(n_steps)]

    def copy(self):
        critic = self.__class__(len(self.core[State.UP]))
        critic.learning_rate_scheme = self.learning_rate_scheme.copy()
        critic.core = {key : self.core[key].copy() for key in self.core.keys()}

        return critic


    def step_evals(self, states, actions):
        evals = [0. for _ in range(len(states))]
        for step_id in range(len(states)):
            state = states[step_id]
            action = actions[step_id]
            state_evals = self.core[state]
            evals[step_id] = state_evals[step_id]
        return evals

    def update(self, states, actions, rewards):
        n_steps = len(states)

        step_evals = self.step_evals(states, actions)

        learning_rates = self.learning_rate_scheme.learning_rates(states, actions)

        self.core[states[0]][0] += (
            learning_rates[0]
            * (
                - step_evals[0]
            )
        )

        for step_id in range(1, n_steps):
            self.core[states[step_id]][step_id] += (
                learning_rates[step_id]
                * (
                    rewards[step_id - 1]
                    + step_evals[step_id - 1]
                    - step_evals[step_id]
                )

            )


class ABaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, states, actions, rewards):
        self.v_critic.update(states, actions, rewards)
        self.q_critic.update(states, actions, rewards)

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions))

    def step_evals(self, states, actions):
        q_step_evals = self.q_critic.step_evals(states, actions)
        v_step_evals = self.v_critic.step_evals(states, actions)
        return [q_step_evals[i] - v_step_evals[i] for i in range(len(q_step_evals))]

class ATrajCritic(ABaseCritic):
    def __init__(self):
        self.v_critic = UTrajCritic()
        self.q_critic = QTrajCritic()

    def copy(self):
        critic = self.__class__()
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

class ASteppedCritic(ABaseCritic):
    def __init__(self, n_steps):
        self.v_critic = USteppedCritic(n_steps)
        self.q_critic = QSteppedCritic(n_steps)
        self.n_steps = n_steps

    def copy(self):
        critic = self.__class__(self.n_steps)
        critic.v_critic = self.v_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic

class UqBaseCritic():
    def __init__(self):
        raise NotImplementedError("Abstract Method")

    def update(self, states, actions, rewards):
        self.u_critic.update(states, actions, rewards)
        self.q_critic.update(states, actions, rewards)

    def eval(self, states, actions):
        return list_sum(self.step_evals(states, actions)) / len(states)

    def step_evals(self, states, actions):
        q_step_evals = self.q_critic.step_evals(states, actions)
        u_step_evals = self.u_critic.step_evals(states, actions)
        return [q_step_evals[i] + u_step_evals[i] for i in range(len(q_step_evals))]

class UqTrajCritic(UqBaseCritic):
    def __init__(self):
        self.u_critic = UTrajCritic()
        self.q_critic = QTrajCritic()

    def copy(self):
        critic = self.__class__()
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic


class UqSteppedCritic(UqBaseCritic):
    def __init__(self, n_steps):
        self.u_critic = USteppedCritic(n_steps)
        self.q_critic = QSteppedCritic(n_steps)
        self.n_steps = n_steps

    def copy(self):
        critic = self.__class__(self.n_steps)
        critic.u_critic = self.u_critic.copy()
        critic.q_critic = self.q_critic.copy()

        return critic



class Domain:
    def __init__(self):
        self.n_steps = 50
        self.max_score = 0
        self.reward_home_a = 0.0
        self.reward_home_b = 10.
        self.reward_up_a = 1.
        self.reward_down_a = 0.0
        self.return_home_prob = 0.2
        self.domain_noise = 0.0

        self.R_expected = np.zeros((2, 3))
        self.R_expected[int(Action.A), int(State.HOME)] = self.reward_home_a
        self.R_expected[int(Action.B), int(State.HOME)] = self.reward_home_b
        self.R_expected[int(Action.A), int(State.UP)] = self.reward_up_a
        self.R_expected[int(Action.B), int(State.UP)] = 0.
        self.R_expected[int(Action.A), int(State.DOWN)] = self.reward_down_a
        self.R_expected[int(Action.B), int(State.DOWN)] = 0.

        self.P_transition = np.zeros((2, 3, 3))
        self.P_transition[:, int(State.UP), int(State.HOME)] = (
            self.return_home_prob
        )
        self.P_transition[:, int(State.UP), int(State.UP)] = (
            1. - self.return_home_prob
        )
        self.P_transition[:, int(State.DOWN), int(State.HOME)] = (
            self.return_home_prob
        )
        self.P_transition[:, int(State.DOWN), int(State.DOWN)] = (
            1. - self.return_home_prob
        )
        self.P_transition[int(Action.A), int(State.HOME), int(State.UP)] = 1.
        self.P_transition[int(Action.B), int(State.HOME), int(State.DOWN)] = 1.

    def expected_value(self, policy):
        P_policy = np.zeros((2, 3))
        P_policy[int(Action.A), int(State.HOME)] = policy[State.HOME]
        P_policy[int(Action.B), int(State.HOME)] = 1. - policy[State.HOME]
        P_policy[int(Action.A), int(State.UP)] = policy[State.UP]
        P_policy[int(Action.B), int(State.UP)] = 1. - policy[State.UP]
        P_policy[int(Action.A), int(State.DOWN)] = policy[State.DOWN]
        P_policy[int(Action.B), int(State.DOWN)] = 1. - policy[State.DOWN]

        values = (
            mdp_policy_values(
                P_policy,
                self.P_transition,
                self.R_expected,
                self.n_steps
            )
        )

        return values[int(State.HOME)]

    def max_value(self):

        values = (
            mdp_optimal_values(
                self.P_transition,
                self.R_expected,
                self.n_steps
            )
        )

        return values[int(State.HOME)]

    def manual_expected_value(self, policy):
        Q = {}
        Q[(State.HOME, Action.A)] = [0. for i in range(self.n_steps)]
        Q[(State.UP, Action.A)] = [0. for i in range(self.n_steps)]
        Q[(State.DOWN, Action.A)] = [0. for i in range(self.n_steps)]
        Q[(State.HOME, Action.B)] = [0. for i in range(self.n_steps)]
        Q[(State.UP, Action.B)] = [0. for i in range(self.n_steps)]
        Q[(State.DOWN, Action.B)] = [0. for i in range(self.n_steps)]


        Q[(State.HOME, Action.A)][0] = self.reward_home_a
        Q[(State.HOME, Action.B)][0] = self.reward_home_b
        Q[(State.UP, Action.A)][0] = self.reward_up_a
        Q[(State.UP, Action.B)][0] = 0.
        Q[(State.DOWN, Action.A)][0] = self.reward_down_a
        Q[(State.DOWN, Action.B)][0] = 0.


        for i in range(1, self.n_steps):
            Q[(State.HOME, Action.A)][i] = (
                self.reward_home_a
                + policy[State.UP] * Q[(State.UP, Action.A)][i-1]
                + (1. - policy[State.UP]) * Q[(State.UP, Action.B)][i-1]
            )

            Q[(State.HOME, Action.B)][i] = (
                self.reward_home_b
                + policy[State.DOWN] * Q[(State.DOWN, Action.A)][i-1]
                + (1. - policy[State.DOWN]) * Q[(State.DOWN, Action.B)][i-1]
            )

            Q[(State.UP, Action.A)][i] = (
                self.reward_up_a
                + (
                    self.return_home_prob
                    * policy[State.HOME]
                    * Q[(State.HOME, Action.A)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.HOME])
                    * Q[(State.HOME, Action.B)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.UP]
                    * Q[(State.UP, Action.A)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.UP])
                    * Q[(State.UP, Action.B)][i-1]
                )
            )

            Q[(State.UP, Action.B)][i] = (
                0.0
                + (
                    self.return_home_prob
                    * policy[State.HOME]
                    * Q[(State.HOME, Action.A)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.HOME])
                    * Q[(State.HOME, Action.B)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.UP]
                    * Q[(State.UP, Action.A)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.UP])
                    * Q[(State.UP, Action.B)][i-1]
                )
            )

            Q[(State.DOWN, Action.A)][i] = (
                self.reward_down_a
                + (
                    self.return_home_prob
                    * policy[State.HOME]
                    * Q[(State.HOME, Action.A)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.HOME])
                    * Q[(State.HOME, Action.B)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.DOWN]
                    * Q[(State.DOWN, Action.A)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.DOWN])
                    * Q[(State.DOWN, Action.B)][i-1]
                )
            )

            Q[(State.DOWN, Action.B)][i] = (
                0.
                + (
                    self.return_home_prob
                    * policy[State.HOME]
                    * Q[(State.HOME, Action.A)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.HOME])
                    * Q[(State.HOME, Action.B)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.DOWN]
                    * Q[(State.DOWN, Action.A)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.DOWN])
                    * Q[(State.DOWN, Action.B)][i-1]
                )
            )

        # end for

        return (
            Q[(State.HOME, Action.A)][-1]
            * policy[State.HOME]
            + Q[(State.HOME, Action.B)][-1]
            * (1. - policy[State.HOME])
        )

    def execute(self, policy):
        state = State.HOME
        rewards = [0. for i in range(self.n_steps)]
        states = [None for i in range(self.n_steps)]
        actions = [None for i in range(self.n_steps)]

        for step_id in range(self.n_steps):
            states[step_id] = state

            if random_uniform() < policy[state] :
                action = Action.A
            else:
                action = Action.B
            actions[step_id] = action

            reward_noise = random_normal() * self.domain_noise


            if state == State.HOME:
                if action == Action.A:
                    rewards[step_id] = self.reward_home_a + reward_noise
                    state = State.UP
                else:
                    rewards[step_id] = self.reward_home_b + reward_noise

                    state = State.DOWN

            elif state == State.UP:
                if action == Action.A:
                    rewards[step_id] = self.reward_up_a + reward_noise
                else:
                    rewards[step_id] = 0. + reward_noise

                if random_uniform() < self.return_home_prob:
                    state = State.HOME

            elif state == State.DOWN:
                if action == Action.A:
                    rewards[step_id] = self.reward_down_a + reward_noise
                else:
                    rewards[step_id] = 0. + reward_noise

                if random_uniform() < self.return_home_prob:
                    state = State.HOME

        return states, actions, rewards


def phenotypes_from_population(population):
    phenotypes = [None] * len(population)

    for i in range(len(population)):
        phenotypes[i] = {"policy" : population[i]}

    return phenotypes

def population_from_phenotypes(phenotypes):
    population = [None] * len(phenotypes)

    for i in range(len(phenotypes)):
        population[i] = phenotypes[i]["policy"]

    return population


def new_policy():
    policy = {}
    policy[State.UP] = 0.5
    policy[State.DOWN] = 0.5
    policy[State.HOME] = 0.5
    return policy

def mutant(phenotype, mutation_factor):
    new_policy = phenotype["policy"].copy()
    new_policy[State.UP] += mutation_factor * random_normal()
    new_policy[State.DOWN] +=  mutation_factor * random_normal()
    new_policy[State.HOME] += mutation_factor * random_normal()

    new_policy[State.UP] = min(1., max(0., new_policy[State.UP]))
    new_policy[State.DOWN] = min(1., max(0., new_policy[State.DOWN]))
    new_policy[State.HOME] = min(1., max(0., new_policy[State.HOME]))

    return {"policy" : new_policy}


def binary_tornament(phenotypes, mutation_factor):
    shuffle(phenotypes)

    new_phenotypes = []

    for i in range(len(phenotypes) // 2):
        if phenotypes[i]["fitness"] > phenotypes[i+1]["fitness"]:
            new_phenotypes.append(phenotypes[i])
            new_phenotypes.append(mutant(phenotypes[i], mutation_factor))
        else:
            new_phenotypes.append(phenotypes[i+1])
            new_phenotypes.append(mutant(phenotypes[i+1], mutation_factor))

    return new_phenotypes