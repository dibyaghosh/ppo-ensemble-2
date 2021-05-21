import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, FixedCategorical, FixedNormal, MaxNormal
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.action_dim = action_space.n if action_space.__class__.__name__ == "Discrete" else action_space.shape[0]
        self.base = base(obs_shape, action_dim=self.action_dim, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def extra_info_template(self):
        return dict()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def get_init_rnn_hxs(self, n_trajs, device):
        return torch.zeros(
            n_trajs, self.recurrent_hidden_state_size, device=device)

    def get_dist(self, inputs, rnn_hxs, masks, last_actions):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, last_actions)
        dist = self.dist(actor_features)
        return dist, rnn_hxs

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, last_actions, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, last_actions)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()
        extra_info = self.extra_info_template.copy()
        return value, action, action_log_probs, rnn_hxs, extra_info

    def get_value(self, inputs, rnn_hxs, masks, last_actions):
        value, _, _ = self.base(inputs, rnn_hxs, masks, last_actions)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, last_actions, extra_info):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, last_actions)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        aux_loss = torch.zeros_like(dist_entropy)
        return value, action_log_probs, dist_entropy, aux_loss, rnn_hxs

class CombinedActorCritic:
    def __init__(self, actor_critics, avg=False):
        self.actor_critics = actor_critics
        self.avg = avg
        self.discrete = self.actor_critics[0].discrete
        self.recurrent = self.is_recurrent
    
    def get_dist_params(self, inputs, all_rnn_hxs, masks, last_actions):
        if all_rnn_hxs is None:
            all_rnn_hxs = [None] * len(self.actor_critics)
        new_rnn_hxs = []
        dists = []
        with torch.no_grad():
            for e, ac in enumerate(self.actor_critics):
                dist, rnn_hxs = ac.get_dist(inputs, all_rnn_hxs[e], masks, last_actions)
                dists.append(dist)
                new_rnn_hxs.append(rnn_hxs)
            
            if self.discrete:
                params = torch.stack([dist.probs for dist in dists], dim=1)
            else:
                mus = torch.stack([dist.mean for dist in dists], dim=1)
                scales = torch.stack([dist.stddev for dist in dists], dim=1)
                params = torch.stack([mus, scales], dim=1)
        return params, new_rnn_hxs

    def get_init_rnn_hxs(self, n_trajs, device):
        return [
        torch.zeros(
            n_trajs, actor_critic.recurrent_hidden_state_size, device=device)
        for actor_critic in self.actor_critics
        ]
    
    def __getattr__(self, attr):
        f = getattr(self.actor_critics[0], attr)
        if callable(f):
            def g(*args, **kwargs):
                print('Calling ', attr)
                return [getattr(actor_critic, attr)(*args, **kwargs) for actor_critic in self.actor_critics][0]
            return g
        print('Getting ', attr)
        return f
    
    def act(self, inputs, rnn_hxs, masks, last_actions, deterministic=False):        
        if not self.recurrent:
            if self.discrete and not self.avg: # MAX DISCRETE
                all_probs, _ = self.get_dist_params(inputs, None, masks, last_actions)
                probs = all_probs.max(dim=1)[0]
                dist = FixedCategorical(probs=probs)
                action = dist.sample()
                action_log_dist = dist.logits
                return 0, action, action_log_dist, rnn_hxs, dict()
            else:
                idx = np.random.choice(len(self.actor_critics))
                return self.actor_critics[idx].act(inputs, rnn_hxs, masks, deterministic, last_actions)
        else:
            raise NotImplementedError()    

    
class DistPolicy(Policy):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, n_ensemble=4, avg=False):
        super().__init__(obs_shape, action_space, base, base_kwargs)
        self.discrete = (action_space.__class__.__name__ == "Discrete")
        self.n_ensemble = n_ensemble
        self.avg = avg

    def act(self, inputs, rnn_hxs, masks, last_actions, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, last_actions)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()
        if self.discrete:
            extra_info = dict(other_dist_params=dist.probs.unsqueeze(1).repeat([1, self.n_ensemble, 1]))
        else:
            mu = dist.mean.unsqueeze(1).repeat([1, self.n_ensemble, 1])
            std = dist.stddev.unsqueeze(1).repeat([1, self.n_ensemble, 1])
            params = torch.stack([mu, std], dim=1)
            extra_info = dict(other_dist_params=params)
        return value, action, action_log_probs, rnn_hxs, extra_info

    @property
    def extra_info_template(self):
        if self.discrete:
            return dict(other_dist_params=torch.ones(self.n_ensemble, self.action_dim)) # Directly save probs
        else:
            return dict(other_dist_params=torch.ones(2, self.n_ensemble, self.action_dim)) # Have to save \mu, \sigma for all ensemble members if max

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, last_actions, extra_info):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, last_actions)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        # Computing aux_loss
        if self.discrete:
            all_other_probs = extra_info['other_dist_params']
            if not self.avg: # MAX
                other_probs, _ = torch.max(all_other_probs, dim=1)
            else:
                other_probs = torch.mean(all_other_probs, dim=1)
            other_dist = FixedCategorical(probs=other_probs)
            if np.random.rand() < 0.02:
                print(dist.kl(other_dist).mean(), np.log(self.num_actions) - dist.entropy().mean(), torch.max(other_dist.probs), torch.max(dist.probs))
            aux_loss = dist.kl(other_dist).mean()
        else:
            other_dist_params = extra_info['other_dist_params']
            other_mu, other_scale = other_dist_params[:, 0], other_dist_params[:, 1]
            other_dist = MaxNormal(other_mu, other_scale, avg=self.avg)
            kls = dist.kl(other_dist)
            aux_loss = kls.mean()
            if np.random.rand() < 0.005:
                print(f'Mean: {kls.mean()}, Max: {kls.max()}, Min: {kls.min()}')
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, aux_loss, rnn_hxs

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, action_dim):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size + action_dim, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks, last_actions):
        x = torch.cat([x, last_actions], dim=-1)

        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, image_shape, recurrent=False, hidden_size=512, action_dim=None):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size, action_dim)
        num_inputs = image_shape[0]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, last_actions):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks, last_actions)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=64, action_dim=None):
        num_inputs = input_shape[0]
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size, action_dim)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, last_actions):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks, last_actions)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
