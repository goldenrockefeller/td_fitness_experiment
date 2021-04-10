cimport cython


@cython.locals(
    val = double,
    d = double
)
cpdef double list_sum(list my_list) except *

@cython.locals(
    new_list = list,
    id = Py_ssize_t,
    d = double
)
cpdef list_multiply(list my_list, double val)

cdef random_cache_size
cdef Py_ssize_t random_uniform_counter
cdef double[::1] np_random_uniform_cache

@cython.locals(
    val = double
)
cpdef double random_uniform() except *


cdef public Py_ssize_t  random_normal_counter
cdef double[::1]  np_random_normal_cache

@cython.locals(
    val = double
)
cpdef double random_normal() except *


cdef class StateEnum:
    cdef public Py_ssize_t UP
    cdef public Py_ssize_t HOME
    cdef public Py_ssize_t DOWN


cdef StateEnum State

cdef class ActionEnum:
    cdef public Py_ssize_t A
    cdef public Py_ssize_t B

cdef ActionEnum Action

cdef class BasicLearningRateScheme():
    cdef dict __dict__
    cdef public double learning_rate

    cpdef list learning_rates(self, list states, list actions)

cdef class ReducedLearningRateScheme():
    cdef dict __dict__
    cdef public double learning_rate

    @cython.locals(
        n_steps = Py_ssize_t
    )
    cpdef list learning_rates(self, list states, list actions)
#
# class TrajLearningRateScheme(LearningRateScheme):
#     cdef dict __dict__
#     cdef public dict denoms
#     cdef public double epsilon
#     cdef public double time_horizon
#     cdef public double rate_boost
#
#
#     @cython.locals(
#         visitation = dict,
#         rates = list,
#         denoms_prev = dict,
#         state = Py_ssize_t
#         action = Py_ssize_t
#
#     )
#     cpdef list learning_rates(self, list states, list actions)
#     def learning_rates(self, states, actions):
#         rates = [0. for _ in range(len(states))]
#         denoms_prev = self.denoms.copy()
#         visitation = {key: 0. for key in self.denoms}
#
#         for state, action in zip(states, actions):
#             visitation[(state, action)] += 1.
#
#         for key in self.denoms:
#             self.denoms[key] += visitation[(state, action)] * visitation[(state, action)]
#
#
#         for step_id, (state, action) in enumerate(zip(states, actions)):
#             rates[step_id] = (
#                 self.rate_boost
#                 / (self.denoms[(state, action)] + self.epsilon)
#             )
#
#
#         # for key in self.denoms:
#         #     self.denoms[key] *= 1. - 1./self.time_horizon
#
#         total_relative_denom = 0.
#
#         for key in self.denoms:
#             total_relative_denom += (
#                 visitation[(state, action)] * visitation[(state, action)]
#                 / (self.denoms[key] + self.epsilon)
#             )
#
#
#         for step_id, key in enumerate(zip(states, actions)):
#             relative_denom = (
#                 visitation[(state, action)] * visitation[(state, action)]
#                 / (self.denoms[key] + self.epsilon)
#             )
#             time_share = relative_denom / (total_relative_denom + self.epsilon)
#
#             rates[step_id] *= time_share
#
#
#
#         for key in self.denoms:
#             relative_denom = (
#                 visitation[(state, action)] * visitation[(state, action)]
#                 / (self.denoms[key] + self.epsilon)
#             )
#             time_share = relative_denom / (total_relative_denom + self.epsilon)
#
#             self.denoms[key] *= 1. - time_share/self.time_horizon
#
#         return rates
#
cdef class SteppedLearningRateScheme():
    cdef dict __dict__
    cdef public dict denoms
    cdef public double time_horizon


    @cython.locals(
        rates = list,
        step_id = Py_ssize_t
    )
    cpdef list learning_rates(self, list states, list actions)

cdef class TrajCritic():
    cdef dict __dict__
    cdef public dict core
    cdef public object learning_rate_scheme


    cpdef double eval(self, list states, list actions) except *

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t
    )
    cpdef list step_evals(self, list states, list actions)

cdef class SteppedCritic():
    cdef dict __dict__
    cdef public dict core
    cdef public object learning_rate_scheme
    cdef public Py_ssize_t n_steps


    cpdef double eval(self, list states, list actions) except *

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
        state_evals = list
    )
    cpdef list step_evals(self, list states, list actions)

cdef class AveragedTrajCritic(TrajCritic):
    pass


cdef class AveragedSteppedCritic(SteppedCritic):
    pass


cdef class MidTrajCritic(TrajCritic):


    @cython.locals(
        n_steps = Py_ssize_t,
        fitness = double,
        estimate = double,
        error = double,
        learning_rates = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
        delta = double
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class MidSteppedCritic(SteppedCritic):

    @cython.locals(
        n_steps = Py_ssize_t,
        fitness = double,
        estimate = double,
        error = double,
        learning_rates = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
        delta = double
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class InexactMidTrajCritic(AveragedTrajCritic):

    @cython.locals(
        n_steps = Py_ssize_t,
        fitness = double,
        estimate = double,
        step_evals = list,
        error = double,
        learning_rates = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
        delta = double
    )
    cpdef update(self, list states, list actions, list rewards)


cdef class InexactMidSteppedCritic(AveragedSteppedCritic):
    @cython.locals(
        n_steps = Py_ssize_t,
        fitness = double,
        estimate = double,
        step_evals = list,
        error = double,
        learning_rates = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
        delta = double
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class QTrajCritic(AveragedTrajCritic):

    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)


cdef class QSteppedCritic(AveragedSteppedCritic):


    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class UTrajCritic(AveragedTrajCritic):

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
    )
    cpdef list step_evals(self, list states, list actions)

    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class USteppedCritic(AveragedSteppedCritic):

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state_evals = list,
        state = Py_ssize_t,
        action = Py_ssize_t,
    )
    cpdef list step_evals(self, list states, list actions)

    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class VTrajCritic(AveragedTrajCritic):

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state = Py_ssize_t,
        action = Py_ssize_t,
    )
    cpdef list step_evals(self, list states, list actions)

    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)

cdef class VSteppedCritic(AveragedSteppedCritic):

    @cython.locals(
        evals = list,
        step_id = Py_ssize_t,
        state_evals = list,
        state = Py_ssize_t,
        action = Py_ssize_t,
    )
    cpdef list step_evals(self, list states, list actions)

    @cython.locals(
        n_steps = Py_ssize_t,
        step_evals = list,
        learning_rates = list,
        step_id = Py_ssize_t
    )
    cpdef update(self, list states, list actions, list rewards)


cdef class ABaseCritic():
    cdef dict __dict__
    cdef public object v_critic
    cdef public object q_critic

    cpdef update(self, list states, list actions, list rewards)

    cpdef double eval(self, list states, list actions) except *

    @cython.locals(
        q_step_evals = list,
        v_step_evals = list,
        i = Py_ssize_t
    )
    cpdef list step_evals(self, list states, list actions)

cdef class ATrajCritic(ABaseCritic):
    pass


cdef class ASteppedCritic(ABaseCritic):
    cdef Py_ssize_t n_steps

cdef class UqBaseCritic():
    cdef dict __dict__
    cdef public object u_critic
    cdef public object q_critic

    cpdef update(self, list states, list actions, list rewards)

    cpdef double eval(self, list states, list actions)  except *

    @cython.locals(
        q_step_evals = list,
        u_step_evals = list,
        i = Py_ssize_t
    )
    cpdef list step_evals(self, list states, list actions)

cdef class UqTrajCritic(UqBaseCritic):
    pass


cdef class UqSteppedCritic(UqBaseCritic):
    cdef Py_ssize_t n_steps

@cython.locals(i = Py_ssize_t)
cpdef list phenotypes_from_population(list population)

@cython.locals(i = Py_ssize_t)
cpdef population_from_phenotypes(list phenotypes)


cdef class Domain:
    cdef dict __dict__
    cdef public Py_ssize_t n_steps
    cdef public double reward_home_a
    cdef public double reward_home_b
    cdef public double reward_up_a
    cdef public double reward_down_a
    cdef public double return_home_prob
    cdef public double domain_noise

    @cython.locals(
        state = Py_ssize_t,
        rewards = list,
        states = list,
        actions = list,
        step_id = Py_ssize_t,
        reward_noise = double
    )
    cpdef tuple execute(self, dict policy)

@cython.locals(
    policy = dict
)
cpdef dict new_policy()

@cython.locals(
    new_policy = dict
)
cpdef dict mutant(dict phenotype, double mutation_factor)

@cython.locals(
    new_phenotypes = list,
    i = Py_ssize_t
)
cpdef list binary_tornament(list phenotypes, double mutation_factor)


