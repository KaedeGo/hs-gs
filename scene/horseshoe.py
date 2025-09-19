import numpy as np
import torch
import math
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import torch.nn.functional as F

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

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            n_inputs, n_outputs = self.mean.shape
        else:
            n_inputs = len(self.mean)
            n_outputs = 1

        part1 = ((n_inputs * n_outputs) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)).cuda()
        part2 = torch.sum(torch.log(self.std_dev))

        return part1 + part2


class InverseGamma(Distribution):
    """ Inverse Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters of the distribution.

        Args:
            shape: torch tensor of floats, shape parameters of the distribution
            rate: torch tensor of floats, rate parameters of the distribution
        """
        self.shape = shape
        self.rate = rate

    def exp_inverse(self):
        """
        Calculates the expectation E[1/x], where x follows
        the inverse gamma distribution
        """
        return self.shape / self.rate

    def exp_log(self):
        """
        Calculates the expectation E[log(x)], where x follows
        the inverse gamma distribution
        """
        exp_log = torch.log(self.rate) - torch.digamma(self.shape)
        return exp_log

    def entropy(self):
        """
        Calculates the entropy of the inverse gamma distribution
        """
        entropy =  self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                     - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def update(self, shape, rate):
        """
        Updates shape and rate of the distribution

        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        """
        self.shape = shape
        self.rate = rate

    def rsample(self, sample_shape):
        return torch.distributions.InverseGamma(self.shape, self.rate).rsample(sample_shape)


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
        self.prior_lambda_shape = torch.Tensor([0.5])
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
        self.beta = ReparameterizedGaussian(self.beta_mean, self.beta_rho)

        # local shrinkage parameters
        self.lambda_shape = nn.Parameter(self.prior_lambda_shape * torch.ones(scaling_matrix.shape)) # (N, 3)
        self.lambda_rate = nn.Parameter(self.prior_lambda_rate * torch.ones(scaling_matrix.shape)) # (N, 3)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        self.theta_shape = nn.Parameter(self.prior_theta_shape) # (1)
        self.theta_rate = nn.Parameter(self.prior_theta_rate) # (1)
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        self.scaling_matrix = scaling_matrix # (N, 3)


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
            exp_gaus = - 0.5 * dim * (math.log(2 * math.pi)) - 0.5 * (torch.sum(mean ** 2) + torch.sum(std ** 2))
            return exp_gaus

        device = self.get_device()

        # E_q[ln p(\lambda)] for the weights
        exp_lambda_inverse = self.lambda_.exp_inverse().to(device)
        exp_log_lambda = self.lambda_.exp_log().to(device)
        shape = self.prior_lambda_shape.to(device)
        rate = self.prior_lambda_rate.to(device)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        shape = self.prior_theta_shape.to(device)
        rate = self.prior_theta_rate.to(device)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, rate, torch.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev)

        return log_gaussian + log_inv_gammas # horseshoe prior
        # return log_gaussian # gaussian prior

    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.
        The entropy is defined as -integral[q(theta) log(q(theta))]. The log of the
        variational posterior is given by integral[q(theta) log(q(theta))].
        Therefore, we compute the entropy and return -entropy.
        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """
        entropy = self.beta.entropy() + self.lambda_.entropy() + self.theta.entropy() # horseshoe prior
        # entropy = self.beta.entropy() # gaussian prior

        return -entropy

    def kl_loss(self):
        return self.log_variational_posterior() - self.log_prior()
            
    
    def sample(self, n_samples=10): # TODO: change n_samples from [1, 2, 4, 8, 16, 32 ...]
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)
        log_lambda = self.lambda_.rsample(sample_shape=(n_samples, )).clamp(min=1e-4, max=1).mean(0)
        log_theta = self.theta.rsample(sample_shape=(n_samples, )).clamp(min=1e-4, max=1).mean(0)

        # TODO calculate the mean of sampled log_lambda and log_theta

        scaling = self.scaling_matrix * log_lambda * log_theta # horseshoe sample

        # sd = torch.nn.functional.softplus(self.beta_rho) # gaussian sample
        # eps = torch.distributions.Normal(0, sd).rsample(sample_shape=(n_samples, )).mean(0)
        # scaling = self.scaling_matrix + eps

        # scaling_sample = torch.distributions.Normal(self.scaling_matrix, sd).rsample(sample_shape=(n_samples, ))
        return scaling # shape: (N, 3)

    def get_device(self):

        return self.lambda_shape.device

    def reset_na(self):

        for param in self.parameters():
            param.data = torch.where(
                torch.isnan(param.data),
                torch.full_like(param.data, 0.5),
                param.data
            )