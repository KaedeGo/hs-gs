import numpy as np
import torch
import math
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F

from torch.distributions import HalfCauchy, InverseGamma

class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

class ReparameterizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        return self.mean + self.std_dev * epsilon

    def log_prob(self):
        pass


def horseshoe_logprior(x, scale=1.0):
    # x is assumed to be positive (diagonal scaling factors)
    # Half-Cauchy density: p(x) ∝ 2/(π * scale * (1 + (x/scale)^2))
    # Log density (up to an additive constant):
    return torch.sum(torch.log(2.) - math.log(math.pi) - torch.log(torch.tensor(scale)) -
                     torch.log(1 + (x/scale)**2))

class Horseshoe(nn.Module):
    """
    Single linear layer of a horseshoe prior for regression
    """
    def __init__(self, scaling_matrix, priors):
        """
        Args
            priors: instance of class HorseshoeHyperparameters
        """
        super().__init__()

        # Scale to initialize weights, according to Yingzhen's work
        scale = priors["horseshoe_scale"]

        # Initialization of parameters of prior distribution
        # weight parameters
        self.prior_tau_shape = torch.Tensor([0.5])

        # local shrinkage parameters
        self.prior_lambda_shape = torch.Tensor([3])
        self.prior_lambda_rate = torch.Tensor([1 / priors["weight_cauchy_scale"] ** 2]) # (1)
        # self.prior_lambda_rate = nn.Parameter(torch.Tensor([1 / priors["weight_cauchy_scale"] ** 2])) # (1)

        # global shrinkage parameters
        # self.prior_v_shape = torch.Tensor([0.5])
        self.prior_theta_shape = torch.Tensor([0.5])
        self.prior_theta_rate = torch.Tensor([1 / priors["global_cauchy_scale"] ** 2])

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mean = nn.Parameter(torch.Tensor(scaling_matrix.shape).uniform_(-scale, scale)) # (N, 3)
        self.beta_rho = nn.Parameter(torch.ones(scaling_matrix.shape) * priors["beta_rho_scale"]) # (N, 3)

        # local shrinkage parameters
        self.lambda_shape = nn.Parameter(self.prior_lambda_shape * torch.ones(scaling_matrix.shape)) # (N, 3)
        self.lambda_rate = nn.Parameter(self.prior_lambda_rate * torch.ones(scaling_matrix.shape)) # (N, 3)
        # self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        # global shrinkage parameters
        # self.theta_shape_raw = nn.Parameter(self.prior_theta_shape)
        # self.theta_rate_raw = nn.Parameter(self.prior_theta_rate)

        self.theta_shape = nn.Parameter(self.prior_theta_shape) # (1)
        self.theta_rate = nn.Parameter(self.prior_theta_rate) # (1)
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        self.scaling_matrix = scaling_matrix # (N, 3)

    # @property
    # def theta_shape(self):
    #     return self.theta_shape_raw.clamp(min=1e-3, max=10)

    # @property
    # def theta_rate(self):
    #     return self.theta_rate_raw.clamp(min=1e-3, max=10)


    def log_prior(self):
        """
        Computes the expectation of the log of the prior p under the variational posterior q
        """
        def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
            exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x \
                      - exp_rate * exp_x_inverse
            return torch.sum(exp_log)

        def exp_log_gaussian(mean, std):
            dim = mean.numel()
            exp_gaus = - 0.5 * dim * (torch.log(2 * torch.pi)) - 0.5 * (torch.sum(mean ** 2) + torch.sum(std ** 2))
            return exp_gaus

        # Calculate E_q[ln p(\tau | \lambda)] + E[ln p(\lambda)]
        # E_q[ln p(\tau | \lambda)] for the weights
        device = self.get_device()

        shape = self.prior_tau_shape.to(device)
        exp_lambda_inverse = self.lambda_.exp_inverse().to(device)
        exp_log_lambda = self.lambda_.exp_log().to(device)
        exp_log_tau = self.log_tau.mean
        exp_tau_inverse = torch.exp(-self.log_tau.mean + 0.5 * self.log_tau.std_dev **2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for the weights
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        # E_q[ln p(v | \theta)] for the global shrinkage parameter
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mean
        exp_v_inverse = torch.exp(-self.log_v.mean + 0.5 * self.log_v.std_dev **2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                exp_log_v, exp_v_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev)

        return log_gaussian + log_inv_gammas

    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.
        The entropy is defined as -integral[q(theta) log(q(theta))]. The log of the
        variational posterior is given by integral[q(theta) log(q(theta))].
        Therefore, we compute the entropy and return -entropy.
        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """
        entropy = self.beta.entropy()\
                + self.log_tau.entropy() + torch.sum(self.log_tau.mean)\
                + self.lambda_.entropy() + self.bias.entropy()\
                + self.log_v.entropy() + torch.sum(self.log_v.mean)\
                + self.theta.entropy()

        if sum(torch.isnan(entropy)).item() != 0:
            raise Exception("entropy/log_variational_posterior computation ran into nan!")
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mean)
            print('beta std: ', self.beta.std_dev)

        return -entropy

    def forward(self):
        assert not torch.isnan(self.beta_mean).any(), "beta_mean contains nan!"
        assert not torch.isnan(self.beta_rho).any(), "beta_rho contains nan!"
        assert not torch.isnan(self.lambda_shape).any(), "lambda_shape contains nan!"
        assert not torch.isnan(self.lambda_rate).any(), "lambda_rate contains nan!"
        assert not torch.isnan(self.theta_shape).any(), "theta_shape contains nan!"
        assert not torch.isnan(self.theta_rate).any(), "theta_rate contains nan!"
        self.safe_lambda_shape = self.lambda_shape.clamp(min=1e-2, max=2)
        self.safe_lambda_rate = self.lambda_rate.clamp(min=1e-2, max=2)
        self.lambda_ = InverseGamma(self.safe_lambda_shape, self.safe_lambda_rate)
        self.safe_theta_shape = self.theta_shape.clamp(min=1e-2, max=2)
        self.safe_theta_rate = self.theta_rate.clamp(min=1e-2, max=2)
        self.theta = InverseGamma(self.safe_theta_shape, self.safe_theta_rate)

        for i in range(10):
            log_lambda = self.lambda_.rsample().clamp(min=1e-4, max=1)
            log_theta = self.theta.rsample().clamp(min=1e-4, max=1)
            sd = torch.nn.functional.softplus(self.beta_rho)
            # sd = torch.log1p(torch.exp(self.beta_rho))
            scale = (sd * log_lambda * log_theta).clamp(min=1e-4)
            if not torch.isnan(scale).any():
                log_prob = torch.distributions.Normal(self.beta_mean, scale).log_prob(self.scaling_matrix)
                return - log_prob.mean()
            else:
                if i < 9:
                    continue
                else:
                    raise Exception("log_prob computation ran into nan!")
            # if torch.isnan(log_prob).any() and i < 9:
            #     continue
            # elif torch.isnan(log_prob).any() and i == 9:
            #     raise Exception("log_prob computation ran into nan!")
            # else:
            #     break
            
    
    def sample(self, n_samples=1):
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)
        log_lambda = self.lambda_.rsample().clamp(min=1e-4, max=1)
        log_theta = self.theta.rsample().clamp(min=1e-4, max=1)
        sd = torch.nn.functional.softplus(self.beta_rho)
        eps = torch.distributions.Normal(0, sd * log_lambda * log_theta).rsample(sample_shape=(n_samples, ))
        scaling = self.scaling_matrix + eps
        # scaling_sample = torch.distributions.Normal(self.scaling_matrix, sd * log_lambda * log_theta).rsample(sample_shape=(n_samples, ))
        return scaling # shape: (N, 3)