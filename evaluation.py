import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device, deterministic=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = actor_critic.get_init_rnn_hxs(num_processes, device) #torch.zeros(
        # num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # Last Actions
    discrete_action = hasattr(eval_envs.action_space, 'n')
    action_dim = eval_envs.action_space.n if discrete_action else eval_envs.action_space.shape[0]
    last_actions = torch.zeros(num_processes, action_dim, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states, _ = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                last_actions,
                deterministic=deterministic)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        if discrete_action:
            last_actions.zero_()
            last_actions[torch.arange(num_processes), action] = 1
        else:
            last_actions.copy_(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return eval_episode_rewards
