
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

from min_entropy_dist import my_optimize
from critic import *


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
        self.A = 0
        self.B = 1
        self.C = 2

State =  StateEnum()

class ActionEnum:
    def __init__(self):
        self.U = 0
        self.V = 1

Action = ActionEnum()

def all_observations():
    yield State.B
    yield State.A
    yield State.C

def all_actions():
    yield Action.U
    yield Action.V


class Domain:
    def __init__(self):
        self.n_steps = 100
        self.max_score = 0
        self.reward_a_u = 5.
        self.reward_a_v = 0.
        self.reward_b_u = 0.
        self.reward_c_u = 1.
        self.return_home_prob = 0.2
        self.domain_noise = 0.0

        self.R_expected = np.zeros((2, 3))
        self.R_expected[int(Action.U), int(State.A)] = self.reward_a_u
        self.R_expected[int(Action.V), int(State.A)] = self.reward_a_v
        self.R_expected[int(Action.U), int(State.B)] = self.reward_b_u
        self.R_expected[int(Action.V), int(State.B)] = 0.
        self.R_expected[int(Action.U), int(State.C)] = self.reward_c_u
        self.R_expected[int(Action.V), int(State.C)] = 0.

        self.P_transition = np.zeros((2, 3, 3))
        self.P_transition[:, int(State.B), int(State.A)] = (
            self.return_home_prob
        )
        self.P_transition[:, int(State.B), int(State.B)] = (
            1. - self.return_home_prob
        )
        self.P_transition[:, int(State.C), int(State.A)] = (
            self.return_home_prob
        )
        self.P_transition[:, int(State.C), int(State.C)] = (
            1. - self.return_home_prob
        )
        self.P_transition[int(Action.U), int(State.A), int(State.B)] = 1.
        self.P_transition[int(Action.V), int(State.A), int(State.C)] = 1.

    def expected_value(self, policy):
        P_policy = np.zeros((2, 3))
        P_policy[int(Action.U), int(State.A)] = policy[State.A]
        P_policy[int(Action.V), int(State.A)] = 1. - policy[State.A]
        P_policy[int(Action.U), int(State.B)] = policy[State.B]
        P_policy[int(Action.V), int(State.B)] = 1. - policy[State.B]
        P_policy[int(Action.U), int(State.C)] = policy[State.C]
        P_policy[int(Action.V), int(State.C)] = 1. - policy[State.C]

        values = (
            mdp_policy_values(
                P_policy,
                self.P_transition,
                self.R_expected,
                self.n_steps
            )
        )

        return values[int(State.A)]

    def max_value(self):

        values = (
            mdp_optimal_values(
                self.P_transition,
                self.R_expected,
                self.n_steps
            )
        )

        return values[int(State.A)]

    def manual_expected_value(self, policy):
        Q = {}
        Q[(State.A, Action.U)] = [0. for i in range(self.n_steps)]
        Q[(State.B, Action.U)] = [0. for i in range(self.n_steps)]
        Q[(State.C, Action.U)] = [0. for i in range(self.n_steps)]
        Q[(State.A, Action.V)] = [0. for i in range(self.n_steps)]
        Q[(State.B, Action.V)] = [0. for i in range(self.n_steps)]
        Q[(State.C, Action.V)] = [0. for i in range(self.n_steps)]


        Q[(State.A, Action.U)][0] = self.reward_a_u
        Q[(State.A, Action.V)][0] = self.reward_a_v
        Q[(State.B, Action.U)][0] = self.reward_b_u
        Q[(State.B, Action.V)][0] = 0.
        Q[(State.C, Action.U)][0] = self.reward_c_u
        Q[(State.C, Action.V)][0] = 0.


        for i in range(1, self.n_steps):
            Q[(State.A, Action.U)][i] = (
                self.reward_a_u
                + policy[State.B] * Q[(State.B, Action.U)][i-1]
                + (1. - policy[State.B]) * Q[(State.B, Action.V)][i-1]
            )

            Q[(State.A, Action.V)][i] = (
                self.reward_a_v
                + policy[State.C] * Q[(State.C, Action.U)][i-1]
                + (1. - policy[State.C]) * Q[(State.C, Action.V)][i-1]
            )

            Q[(State.B, Action.U)][i] = (
                self.reward_b_u
                + (
                    self.return_home_prob
                    * policy[State.A]
                    * Q[(State.A, Action.U)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.A])
                    * Q[(State.A, Action.V)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.B]
                    * Q[(State.B, Action.U)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.B])
                    * Q[(State.B, Action.V)][i-1]
                )
            )

            Q[(State.B, Action.V)][i] = (
                0.0
                + (
                    self.return_home_prob
                    * policy[State.A]
                    * Q[(State.A, Action.U)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.A])
                    * Q[(State.A, Action.V)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.B]
                    * Q[(State.B, Action.U)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.B])
                    * Q[(State.B, Action.V)][i-1]
                )
            )

            Q[(State.C, Action.U)][i] = (
                self.reward_c_u
                + (
                    self.return_home_prob
                    * policy[State.A]
                    * Q[(State.A, Action.U)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.A])
                    * Q[(State.A, Action.V)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.C]
                    * Q[(State.C, Action.U)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.C])
                    * Q[(State.C, Action.V)][i-1]
                )
            )

            Q[(State.C, Action.V)][i] = (
                0.
                + (
                    self.return_home_prob
                    * policy[State.A]
                    * Q[(State.A, Action.U)][i-1]
                )
                + (
                    self.return_home_prob
                    * (1. - policy[State.A])
                    * Q[(State.A, Action.V)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * policy[State.C]
                    * Q[(State.C, Action.U)][i-1]
                )
                + (
                    (1. - self.return_home_prob)
                    * (1. - policy[State.C])
                    * Q[(State.C, Action.V)][i-1]
                )
            )

        # end for

        return (
            Q[(State.A, Action.U)][-1]
            * policy[State.A]
            + Q[(State.A, Action.V)][-1]
            * (1. - policy[State.A])
        )

    def execute(self, policy):
        state = State.A
        rewards = [0. for i in range(self.n_steps)]
        states = [None for i in range(self.n_steps)]
        actions = [None for i in range(self.n_steps)]

        for step_id in range(self.n_steps):
            states[step_id] = state

            if random_uniform() < policy[state] :
                action = Action.U
            else:
                action = Action.V
            actions[step_id] = action

            reward_noise = random_normal() * self.domain_noise


            if state == State.A:
                if action == Action.U:
                    rewards[step_id] = self.reward_a_u + reward_noise
                    state = State.B
                else:
                    rewards[step_id] = self.reward_a_v + reward_noise

                    state = State.C

            elif state == State.B:
                if action == Action.U:
                    rewards[step_id] = self.reward_b_u + reward_noise
                else:
                    rewards[step_id] = 0. + reward_noise

                if random_uniform() < self.return_home_prob:
                    state = State.A

            elif state == State.C:
                if action == Action.U:
                    rewards[step_id] = self.reward_c_u + reward_noise
                else:
                    rewards[step_id] = 0. + reward_noise

                if random_uniform() < self.return_home_prob:
                    state = State.A

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


def new_policy_basic():
    policy = {}
    policy[State.B] = 1.0
    policy[State.A] = 1.0
    policy[State.C] = 0.0
    return policy


def new_policy(dist):
    policy = {}
    policy[State.B] = np.random.dirichlet(dist[State.B])[0]
    policy[State.A] = np.random.dirichlet(dist[State.A])[0]
    policy[State.C] = np.random.dirichlet(dist[State.C])[0]
    return policy

# def update_dist(dist, speed, phenotypes):
#
#     for i in range(len(phenotypes) // 2):
#         if phenotypes[i]["fitness"] > phenotypes[i+1]["fitness"]:
#             better_policy = phenotypes[i]["policy"]
#         else:
#             better_policy = phenotypes[i+1]["policy"]
#
#         for state in better_policy.keys():
#             dist[state][0] += speed * better_policy[state]
#             dist[state][1] += speed * (1. - better_policy[state])
#

def update_dist(dist, kl_penalty_factor, phenotypes):
    phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])

    for observation in all_observations():
        data = []
        for i in range(len(phenotypes) // 2):
            datum = np.zeros(2)
            prob_U = phenotypes[i]["policy"][observation]
            prob_V = 1. - prob_U
            datum[0] = prob_U
            datum[1] = prob_V
            data.append(datum)

        dist[observation] = my_optimize(dist[observation], kl_penalty_factor, data)

# def update_dist(dist, kl_penalty_factor, phenotypes):
#
#     phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])
#
#     for i in range(len(phenotypes) // 2):
#         policy = phenotypes[i]["policy"]
#
#         for state in policy.keys():
#             dist[state][0] += speed * policy[state]
#             dist[state][1] += speed * (1. - policy[state])
#
#     for state in policy.keys():
#         dist[state][0] *= sustain
#         dist[state][1] *= sustain

# def update_probs(probs, denoms, phenotypes):
#     new_probs = probs.copy()
#     shuffle(phenotypes)


# def new_policy(probs, parameters):
#     policy = {}
#     p = 100
#     d = 200
#     policy[State.B] = 1.0 if random_uniform() < (probs[State.B] + p) / (denoms + d) else 0.
#     policy[State.A] = 1.0 if random_uniform() <( probs[State.A] + p) / (denoms + d) else 0.
#     policy[State.C] = 1.0 if random_uniform() <( probs[State.C]  + p) / (denoms + d) else 0.
#     return policy
#
# def update_probs(probs, denoms, phenotypes):
#     new_probs = probs.copy()
#     shuffle(phenotypes)
#
#
#     for i in range(len(phenotypes) // 2):
#         if phenotypes[i]["fitness"] > phenotypes[i+1]["fitness"]:
#             better_policy = phenotypes[i]["policy"]
#         else:
#             better_policy = phenotypes[i+1]["policy"]
#
#         for state in better_policy.keys():
#             new_probs[state] += better_policy[state]
#         denoms += 1
#
#     for state in new_probs.keys():
#         new_probs[state] *= 0.99
#
#     denoms *= 0.99
#
#     return new_probs, denoms
#
# # def update_probs(probs, denoms, phenotypes):
# #     new_probs = probs.copy()
# #
# #     phenotypes.sort(reverse = True, key = lambda phenotype : phenotype["fitness"])
# #
# #     for i in range(len(phenotypes) // 2):
# #         policy = phenotypes[i]["policy"]
# #
# #         for state in policy.keys():
# #             new_probs[state] += policy[state]
# #         denoms += 1
# #
# #     for state in new_probs.keys():
# #         new_probs[state] *= 0.99
# #
# #     denoms *= 0.99
# #
# #     return new_probs, denoms
#
# #
# # def mutant(phenotype, mutation_factor):
# #     new_policy = phenotype["policy"].copy()
# #
#     for state in [State.B, State.C, State.A]:
#         if random_uniform() < mutation_factor:
#             if random_uniform() < 0.5:
#                 new_policy[state] = 0.
#             else:
#                 new_policy[state] = 1.
#
#     return {"policy" : new_policy}

#
# def binary_tornament(phenotypes, mutation_factor):
#     shuffle(phenotypes)
#
#     new_phenotypes = []
#     selection_rate = 0.2
#
#     for i in range(len(phenotypes) // 2):
#         if random_uniform() < selection_rate:
#             if phenotypes[i]["fitness"] > phenotypes[i+1]["fitness"]:
#                 new_phenotypes.append(phenotypes[i])
#                 new_phenotypes.append(mutant(phenotypes[i], mutation_factor))
#             else:
#                 new_phenotypes.append(phenotypes[i+1])
#                 new_phenotypes.append(mutant(phenotypes[i+1], mutation_factor))
#         else:
#             if random_uniform() < 0.5:
#                 new_phenotypes.append(phenotypes[i])
#                 new_phenotypes.append(mutant(phenotypes[i], mutation_factor))
#             else:
#                 new_phenotypes.append(phenotypes[i+1])
#                 new_phenotypes.append(mutant(phenotypes[i+1], mutation_factor))
#
#     return new_phenotypes