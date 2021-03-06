import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import DistPolicy, CombinedActorCritic
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import copy

def init_process(rank, size, fn, args, backend='gloo', port='29500'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args, rank, size)

def main(args, rank, size):
    args = copy.deepcopy(args)
    args.save_dir = os.path.join(args.save_dir, f'ensemble_{rank}')

    torch.manual_seed(size * args.seed + rank)
    torch.cuda.manual_seed_all(size * args.seed + rank)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(os.path.expanduser(args.log_dir), f'ensemble_{rank}')
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    # DIST SPECIFIC
    torch.set_num_threads(1)
    if args.diff_gpu:
        n_gpus = torch.cuda.device_count()
        device = torch.device(f"cuda:{rank % n_gpus}" if args.cuda else "cpu")
    else:
        n_gpus = torch.cuda.device_count()
        device = torch.device(f"cuda:0" if args.cuda else "cpu")
    # END DIST SPECIFIC

    envs = make_vec_envs(args.env_name, args.num_processes * (size * args.seed + rank), args.num_processes,
                         args.gamma, args.log_dir, device, False)
    
    # DIST SPECIFIC
    all_actor_critics = [
        DistPolicy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            n_ensemble=size,
            avg=args.avg,
        )
        for e in range(size)
    ]
    [ac.to(device) for ac in all_actor_critics]
    actor_critic = all_actor_critics[rank]
    combined_actor_critic = CombinedActorCritic(all_actor_critics, avg=args.avg)
    # END DIST SPECIFIC

    # actor_critic = Policy(
    #     envs.observation_space.shape,
    #     envs.action_space,
    #     base_kwargs={'recurrent': args.recurrent_policy})
    # actor_critic.to(device)
    print(actor_critic)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.aux_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, extra_info_template=actor_critic.extra_info_template)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    expl_returns = []
    eval_returns = []
    combined_rnn_hxs = combined_actor_critic.get_init_rnn_hxs(args.num_processes, device)

    def save_returns():
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass
        print('Saving Returns to ', save_path)
        np.savez(os.path.join(save_path, 'returns.npz'), exploration=np.array(expl_returns), evaluation=np.array(eval_returns))


    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, extra_info = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], rollouts.last_actions[step])
                params, combined_rnn_hxs = combined_actor_critic.get_dist_params(rollouts.obs[step], combined_rnn_hxs, rollouts.masks[step], rollouts.last_actions[step])
                extra_info['other_dist_params'] = params
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    expl_returns.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, extra_info)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], rollouts.last_actions[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, aux_loss = agent.update(rollouts)

        rollouts.after_update()

        # DIST SPECIFIC
        if (j % args.sync_interval == 0):
            with torch.no_grad():
                for e in range(size):
                    for tensor in all_actor_critics[e].state_dict().values():
                        dist.broadcast(tensor, e)
                        
        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Ensemble {}/{} Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(rank+1, size, j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            new_eval_returns = evaluate(actor_critic, obs_rms, args.eval_env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
            eval_returns.append(new_eval_returns)
            save_returns()
        elif j % 10 == 0:
            save_returns()


            


if __name__ == "__main__":
    args = get_args()
    print(args)
    size = args.n_ensemble
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, main, args, 'gloo', str(args.port)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()