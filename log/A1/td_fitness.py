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
        # prev_V = 1. * V
        # V *= 0
        # for action_id in range(n_actions):
        #     V += (
        #         P_policy[action_id]
        #         * (
        #             np.matmul(P_transition[action_id], prev_V)
        #             + R_expected[action_id]
        #         )
        #     )
        V = np.matmul(P_transition[action_id], V) + R_expected_reduced

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

class TrajLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 10., epsilon = 1.e-9):
        self.denoms = {key: 0. for key in ref_model}
        self.epsilon = epsilon
        self.time_horizon = time_horizon
        self.rate_boost = 10. #replace with nesterov
        # need to use weighted averaging

    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = self.denoms.copy()
        scheme.epsilon = self.epsilon
        scheme.time_horizon = self.time_horizon
        scheme.rate_boost = self.rate_boost

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]
        denoms_prev = self.denoms.copy()
        visitation = {key: 0. for key in self.denoms}

        for state, action in zip(states, actions):
            visitation[(state, action)] += 1.

        for key in self.denoms:
            self.denoms[key] += visitation[(state, action)] * visitation[(state, action)]


        for step_id, (state, action) in enumerate(zip(states, actions)):
            rates[step_id] = (
                self.rate_boost
                / (self.denoms[(state, action)] + self.epsilon)
            )


        # for key in self.denoms:
        #     self.denoms[key] *= 1. - 1./self.time_horizon

        total_relative_denom = 0.

        for key in self.denoms:
            total_relative_denom += (
                visitation[(state, action)] * visitation[(state, action)]
                / (self.denoms[key] + self.epsilon)
            )


        for step_id, key in enumerate(zip(states, actions)):
            relative_denom = (
                visitation[(state, action)] * visitation[(state, action)]
                / (self.denoms[key] + self.epsilon)
            )
            time_share = relative_denom / (total_relative_denom + self.epsilon)

            rates[step_id] *= time_share



        for key in self.denoms:
            relative_denom = (
                visitation[(state, action)] * visitation[(state, action)]
                / (self.denoms[key] + self.epsilon)
            )
            time_share = relative_denom / (total_relative_denom + self.epsilon)

            self.denoms[key] *= 1. - time_share/self.time_horizon

        return rates

class SteppedLearningRateScheme():

    def __init__(self, ref_model, time_horizon = 100.):
        self.denoms = {key: [0. for _ in range(len(ref_model[key]))] for key in ref_model}
        self.time_horizon = time_horizon


    def copy(self):
        scheme = self.__class__(self.denoms)
        scheme.denoms = {key : self.denoms[key].copy() for key in self.denoms}
        scheme.time_horizon = scheme.time_horizon

        return scheme


    def learning_rates(self, states, actions):
        rates = [0. for _ in range(len(states))]


        for step_id, (state, action) in enumerate(zip(states, actions)):
            if (state, action) in self.denoms:
                self.denoms[(state, action)][step_id] += 1
                rates[step_id] = 1. / self.denoms[(state, action)][step_id]
                self.denoms[(state, action)][step_id] *= 1. - 1. / self.time_horizon
            else:
                self.denoms[state][step_id] += 1
                rates[step_id] = 1. / self.denoms[state][step_id]
                self.denoms[state][step_id] *= 1. - 1. / self.time_horizon

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
            delta = error * learning_rates[step_id] / len(states)
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


class Runner:
    def __init__(self, experiment_name, setup_funcs):
        self.setup_funcs = setup_funcs
        self.stat_runs_completed = 0
        self.experiment_name = experiment_name
        setup_names = []
        for setup_func in setup_funcs:
            setup_names.append(setup_func.__name__)
        self.trial_name = "_".join(setup_names)

        # Create experiment folder if not already created.
        try:
            os.makedirs(os.path.join("log", experiment_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # Save experiment details
        filenames_in_folder = (
            glob.glob("./**.py", recursive = True)
            + glob.glob("./**.pyx", recursive = True)
            + glob.glob("./**.pxd", recursive = True))
        for filename in filenames_in_folder:
            copy(filename, os.path.join("log", experiment_name, filename))


    def new_run(self):
        datetime_str = (
            dt.datetime.now().isoformat()
            .replace("-", "").replace(':', '').replace(".", "_")
        )

        print(
            "Starting trial.\n"
            f"experiment: {self.experiment_name}\n"
            f"trial: {self.trial_name}\n"
            f"stat run #: {self.stat_runs_completed}\n"
            "datetime: {datetime_str}\n\n"
            .format(**locals()) )

        args = {}

        for setup_func in self.setup_funcs:
            setup_func(args)

        critic = args["critic"]
        n_steps = args["n_steps"]
        domain_noise = args["domain_noise"]

        domain = Domain()
        domain.domain_noise = domain_noise
        domain.n_steps = n_steps
        n_epochs = 10 * n_steps
        mutation_factor = 0.01
        n_policies = 50

        population = [new_policy() for _ in range(n_policies)]

        n_epochs_elapsed = list(range(1, n_epochs + 1))
        n_training_episodes_elapsed = [n_epochs_elapsed[epoch_id] * n_policies for epoch_id in range(n_epochs)]
        n_training_steps_elapsed = [n_epochs_elapsed[epoch_id] * n_steps * n_policies for epoch_id in range(n_epochs)]
        max_expected_value = domain.max_value()
        max_expected_returns = [max_expected_value for _ in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_evals = []
        critic_score_losses = []
        critic_expected_return_losses = []


        for epoch in range(n_epochs):
            phenotypes = phenotypes_from_population(population)

            new_critic = critic.copy()

            for phenotype in phenotypes:
                policy = phenotype["policy"]
                states, actions, rewards = domain.execute(policy)
                fitness = critic.eval(states, actions)
                new_critic.update(states, actions, rewards)
                phenotype["fitness"] = fitness

            critic = new_critic
            phenotypes = binary_tornament(phenotypes, mutation_factor)
            population = population_from_phenotypes(phenotypes)

            candidate_policy = population[0]
            states, actions, rewards = domain.execute(candidate_policy)
            #print(f"Score: {domain.expected_value(candidate_policy)}")

            score = list_sum(rewards)
            expected_return = domain.expected_value(candidate_policy)
            critic_eval = critic.eval(states, actions)
            critic_score_loss = 0.5 * (critic_eval - score) ** 2
            critic_expected_return_loss = 0.5 * (critic_eval - expected_return) ** 2

            scores.append(score)
            expected_returns.append(expected_return)
            critic_evals.append(  critic.eval(states, actions) )
            critic_score_losses.append( critic_score_loss )
            critic_expected_return_losses.append( critic_expected_return_loss)

            # print(critic.learning_rate_scheme.denoms)
        # end for epoch in range(n_epochs):

        save_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                self.trial_name,
                f"record_{datetime_str}.csv"
            )
        )

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_epochs_elapsed'] + n_epochs_elapsed)
            writer.writerow(['n_training_episodes_elapsed'] + n_training_episodes_elapsed)
            writer.writerow(['n_training_steps_elapsed'] + n_training_steps_elapsed)

            writer.writerow(['max_expected_returns'] + max_expected_returns)
            writer.writerow(['scores'] + scores)
            writer.writerow(['expected_returns'] + expected_returns)
            writer.writerow(['critic_evals'] + critic_evals)
            writer.writerow(['critic_score_losses'] + critic_score_losses)
            writer.writerow(['critic_expected_return_losses'] + critic_expected_return_losses)

        self.stat_runs_completed += 1


        self.critic = critic

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
    args["domain_noise"] = 1.

def mtc(args):
    args["critic"] = MidTrajCritic()

def msc(args):
    args["critic"] = MidSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedLearningRateScheme(args["critic"].core)

def imtc(args):
    args["critic"] = InexactMidTrajCritic()

def imsc(args):
    args["critic"] = InexactMidSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedLearningRateScheme(args["critic"].core)

def qtc(args):
    args["critic"] = QTrajCritic()

def qsc(args):
    args["critic"] = QSteppedCritic(args["n_steps"])
    args["critic"].learning_rate_scheme = SteppedLearningRateScheme(args["critic"].core)

def uqtc(args):
    args["critic"] = UqTrajCritic()

def uqsc(args):
    args["critic"] = UqSteppedCritic(args["n_steps"])
    args["critic"].u_critic.learning_rate_scheme = SteppedLearningRateScheme(args["critic"].u_critic.core)
    args["critic"].q_critic.learning_rate_scheme = SteppedLearningRateScheme(args["critic"].q_critic.core)


def atc(args):
    args["critic"] = ATrajCritic()

def asc(args):
    args["critic"] = ASteppedCritic(args["n_steps"])
    args["critic"].v_critic.learning_rate_scheme = SteppedLearningRateScheme(args["critic"].v_critic.core)
    args["critic"].q_critic.learning_rate_scheme = SteppedLearningRateScheme(args["critic"].q_critic.core)
