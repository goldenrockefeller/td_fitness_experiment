from scipy.special import gamma, psi, polygamma
from numpy import log
from scipy import optimize
from numpy import array, zeros, inf, exp, minimum, maximum, sum,  abs, where, isfinite
from numpy.random import dirichlet
from numpy import finfo

from numba import jit


def dist_entropy(dist):
    return (
        - log(gamma(dist.sum()))
        + log(gamma(dist)).sum()
        + (dist.sum()-len(dist)) * psi(dist.sum())
        - ( (dist - 1.) * psi(dist) ).sum()
    )


def dist_to_dist_entropy(ref_dist, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    ref_dist_sum = ref_dist.sum()
    dist_sum = dist.sum()


    return (
        log(gamma(ref_dist_sum))
        - log(gamma(ref_dist)).sum()
        - log(gamma(dist_sum))
        + log(gamma(dist)).sum()
        + ( (ref_dist - dist) * (psi(ref_dist) - psi(ref_dist_sum)) ).sum()
    )


def data_to_dist_entropy(data, dist):
    return sum([datum_to_dist_entropy(datum, dist) for datum in data]) / len(list(data))


def datum_to_dist_entropy(datum, dist):
    # from https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/

    return (
        log(gamma(dist)).sum()
        - log(gamma(dist.sum()))
        - ((dist - 1.)*log(datum)).sum()
    )


def scorer(ref_dist, kl_penalty_factor, data):
    def score(dist):
        return data_to_dist_entropy(data, dist) + kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score


def min_entropy_dist(ref_dist, kl_penalty_factor, data):
    score = scorer(ref_dist, kl_penalty_factor, data)
    return optimize.minimize(score, ref_dist, method='nelder-mead')


def scorer_2(ref_dist, kl_penalty_factor, dest):
    def score(dist):
        return dist_to_dist_entropy(dest, dist)+ kl_penalty_factor * dist_to_dist_entropy(ref_dist, dist)
    return score


def min_entropy_dist_2(ref_dist, kl_penalty_factor, dest):
    score = scorer_2(ref_dist, kl_penalty_factor, dest)
    return optimize.minimize(score, ref_dist, method='nelder-mead')

def inv_psi(x):
    s = x
    prev_s = -inf
    prev_s_mid = -inf

    s_mid = -inf
    s_hi = inf
    s_lo = -inf
    for i in range(1000):
        prev_s = s
        f = psi(exp(s)) - x
        s = s - (f / (polygamma(1, exp(s)) * exp(s)))



        f = psi(exp(s)) - x

        if isfinite(s).all():
            if (prev_s == s).all():
                return(exp(s))

            if sum(abs(f)) == 0:
                return exp(s)



        s_hi = where(f > 0, minimum(s, s_hi), s_hi)
        s_lo = where(f <= 0, maximum(s, s_lo), s_lo)
        s = where(f > 0, s_hi, s_lo)


        prev_s_mid = s_mid
        s_mid = 0.5 * (s_hi + s_lo)

        if isfinite(s_mid).all() :
            f = psi(exp(s_mid)) - x

            if sum(abs(f)) == 0:
                return exp(s_mid)

            s_hi = where(f > 0, minimum(s_mid, s_hi), s_hi)
            s_lo = where(f <= 0, maximum(s_mid, s_lo), s_lo)
            s_mid = where(f > 0, s_hi, s_lo)

            if (prev_s_mid == s_mid).all():
                return(exp(s_mid))


    return exp(s)


def my_optimize(ref_dist, kl_penalty_factor, data):
    # Using fixed point method http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    dist = ref_dist

    log_pk = 0. * data[0]
    for datum in data:
        log_pk += log(datum + 1e-9)
    log_pk /= len(data)

    log_pk = log_pk / (1 + kl_penalty_factor)

    psi_ref = kl_penalty_factor * (psi(ref_dist) - psi(ref_dist.sum())) / (1 + kl_penalty_factor)

    for i in range(100):
        for param_id in range(len(dist)):
            v = psi(dist.sum()) + log_pk[param_id] + psi_ref[param_id]
            dist[param_id] = inv_psi(v)

    return dist


