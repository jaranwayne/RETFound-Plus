import torch

from lifelines import KaplanMeierFitter
import numpy as np
import random,torch,pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

def loglogistic_activation(mu_sig):
    """
    Activation which ensures mu is between -3 and 3 and sigma is such that
    prediction is not more precise than 1 / n of a year.
    :param mu_logsig:
    :return:
    """
    n = 12  # 1 / n is the fraction of the year in which at least p quantile of the distribution lies
    p = .95  # quantile
    mu = torch.clip(mu_sig[:, 0], -3, 3)
    sig = torch.exp(mu_sig[:, 1]) 
    thrs = torch.log(torch.tensor(1 / (2 * n)) * (torch.exp(-mu) + torch.sqrt((2 * n) ** 2 + torch.exp(-2 * mu)))) / torch.log(torch.tensor(p / (1 - p)))
    log_sig = torch.log(thrs + torch.relu(sig - thrs))

    return mu, log_sig



def logistic_nll_loss( mu, log_sigma, x, d):
    """
    Loss function based on the negative log likelihood of a logistic distribution with
    right censored data.
    :param mu: tensor of the location latent parameter
    :param sigma: tensor of the scale latent parameter
    :param x: tensor of min(log(t), log(c)), t is event time and c is censor time
    :param d: tensor of indicator t < c
    :return:
    """
    x_scaled = (torch.log(x) - mu) / torch.exp(log_sigma)
    nll = x_scaled + d * log_sigma + (1 + d) * torch.log(1 + torch.exp(-x_scaled))

    return torch.mean(nll)

#----------------region of integrated_brier_score----------------

def logistic_survival_fn_gen(location, scale):
    def logistic_survival_fn(t):
        return 1 / (1 + np.exp((np.log(t) - location) / scale))
    return logistic_survival_fn

def brier_score(t, event_times, events_observed, surv_prob_at_t, censor_data=None):

    if censor_data is not None:
        kmf = KaplanMeierFitter().fit(censor_data[0], 1 - censor_data[1])
    else:
        kmf = KaplanMeierFitter().fit(event_times, 1 - events_observed)

    surv_at_t = surv_prob_at_t(t)
    score = 0
    # Uncensored branch
    uncens_cond = np.logical_and(event_times <= t, events_observed)
    uncens_t = event_times[uncens_cond]

    score += np.sum(
        np.float_power((0 - surv_at_t[uncens_cond]), 2) / kmf.survival_function_at_times(uncens_t).values
    )

    # Censored branch
    cens_cond = event_times > t
    score += np.sum(np.float_power(1 - surv_at_t[cens_cond], 2) / kmf.survival_function_at_times(t).iloc[0])

    score /= len(event_times)

    return score


def integrated_brier_score(survival_fn, event_times, events_observed, t_min=None, t_max=None, bins=100, censor_data=None):

    t_min = 0 if t_min is None else max(t_min, 0)
    t_max = max(event_times) if t_max is None else min(t_max, max(event_times))
    t_min += torch.finfo(torch.float32).eps  # Corrections for endpoints
    t_max -= torch.finfo(torch.float32).eps  # Corrections for endpoints
    times = np.linspace(t_min, t_max, bins)

    scores = np.asarray([brier_score(t, event_times, events_observed, survival_fn, censor_data) for t in times])
    ibs = np.trapz(scores, times) / (t_max - t_min)

    return ibs


#----------------concordance_index----------------
def concordance_index(tau, times, pred_times, event_observed=None, censor_data=None):
    """
    Caveat: censor_data times and times need to be the same units
    :param tau: time at which to truncate C-index calculation
    :param times: original times of event
    :param pred_times: or scores. the higher, the lower risk
    :param event_observed: event statuses
    :param censor_data: additional data for KM censor weighing
    :return:
    """
    if event_observed is None:
        event_observed = np.ones((len(times),))

    correct_pairs = 0
    all_pairs = 0

    if censor_data is not None:
        kmf = KaplanMeierFitter().fit(censor_data[0], 1 - censor_data[1])
    else:
        kmf = KaplanMeierFitter().fit(times, 1 - event_observed)

    g = 1 / (kmf.survival_function_at_times(times).values ** 2)

    for j in range(len(times)):
        this_correct_pairs = 1 * g * (times < times[j]) * (times < tau) * (pred_times < pred_times[j])
        this_all_pairs = 1 * g * (times < times[j]) * (times < tau)

        correct_pairs += np.sum(this_correct_pairs[event_observed > 0])
        all_pairs += np.sum(this_all_pairs[event_observed > 0])

    c_idx = correct_pairs / all_pairs

    return c_idx

