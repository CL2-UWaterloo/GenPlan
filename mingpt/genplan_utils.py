import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import math

def sample_actions_log_probs(distribution):
    actions = distribution.sample()
    log_prob_actions = distribution.log_prob(actions)
    return actions, log_prob_actions


# noinspection PyAbstractClass
class CategoricalActionDistribution:
    def __init__(self, raw_logits):
        """
        Ctor.
        :param raw_logits: unprocessed logits, typically an output of a fully-connected layer
        """

        self.raw_logits = raw_logits
        self.log_p = self.p = None

    @property
    def probs(self):
        # if self.p is None:
        self.p = F.softmax(self.raw_logits, dim=-1)
        return self.p

    @property
    def log_probs(self):
        # if self.log_p is None:
        self.log_p = F.log_softmax(self.raw_logits, dim=-1)
        return self.log_p

    def sample_gumbel(self):
        sample = torch.argmax(
            self.raw_logits - torch.empty_like(self.raw_logits).exponential_().log_(),
            -1,
        )
        return sample

    def sample(self):
        # samples = torch.multinomial(self.probs, 1, True).squeeze(dim=-1)
        samples = Categorical(self.probs).sample()
        return samples

    def sample_max(self):
        samples = torch.argmax(self.probs, dim=-1)
        return samples

    def log_prob(self, value):
        value = value.long().unsqueeze(-1)
        log_probs = torch.gather(self.log_probs, -1, value).view(-1)
        return log_probs

    def entropy(self):
        p_log_p = self.log_probs * self.probs
        return -p_log_p.sum(-1)

    def _kl(self, other_log_probs):
        probs, log_probs = self.probs, self.log_probs
        kl = probs * (log_probs - other_log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_inverse(self, other_log_probs):
        probs, log_probs = self.probs, self.log_probs
        kl = torch.exp(other_log_probs) * (other_log_probs - log_probs)
        kl = kl.sum(dim=-1)
        return kl

    def _kl_symmetric(self, other_log_probs):
        return 0.5 * (self._kl(other_log_probs) + self._kl_inverse(other_log_probs))

    def symmetric_kl_with_uniform_prior(self):
        probs, log_probs = self.probs, self.log_probs
        num_categories = log_probs.shape[-1]
        uniform_prob = 1 / num_categories
        log_uniform_prob = math.log(uniform_prob)

        return 0.5 * (
            (probs * (log_probs - log_uniform_prob)).sum(dim=-1)
            + (uniform_prob * (log_uniform_prob - log_probs)).sum(dim=-1)
        )

    def kl_divergence(self, other):
        return self._kl(other.log_probs)
