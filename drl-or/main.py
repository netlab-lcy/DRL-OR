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
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from net_env.simenv import NetEnv


def main():
    # init 
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # for reproducible
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "/eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    ckpt_step = args.ckpt_steps # model save every ckpt_step
    

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # set up environment
    envs = NetEnv(args) 
    num_agent, num_node, observation_spaces, action_spaces, num_type = envs.setup(args.env_name, args.demand_matrix)
    request, obses = envs.reset()
    print("observation_spaces", observation_spaces)

    # open log file
    log_dist_files = []
    log_demand_files = []
    log_delay_files = []
    log_throughput_files = []
    log_loss_files = []
    for i in range(num_type):
        log_dist_file = open("%s/dist_type%d.log" % (log_dir, i), "w", 1)
        log_dist_files.append(log_dist_file)
        log_demand_file = open("%s/demand_type%d.log" % (log_dir, i), "w", 1)
        log_demand_files.append(log_demand_file)
        log_delay_file = open("%s/delay_type%d.log" % (log_dir, i), "w", 1)
        log_delay_files.append(log_delay_file)
        log_throughput_file = open("%s/throughput_type%d.log" % (log_dir, i), "w", 1)
        log_throughput_files.append(log_throughput_file)
        log_loss_file = open("%s/loss_type%d.log" % (log_dir, i), "w", 1)
        log_loss_files.append(log_loss_file)
    log_globalrwd_file = open("%s/globalrwd.log" % (log_dir), "w", 1)
    log_circle_file = open("%s/circle.log" % (log_dir), "w", 1)

    # building model
    actor_critics = []
    agents = []
    rollouts = []
    for i in range(num_agent):
        actor_critic = Policy(observation_spaces[i].shape, action_spaces[i], num_node, num_node, num_type,
            base_kwargs={'recurrent': args.recurrent_policy})
        
        # load parameter
        if model_load_path != None:
            #actor_critic.load_state_dict(torch.load("%s/agent%d.pth" % (model_load_path, i), map_location='cpu')) #gpu data to cpu
            actor_critic.load_state_dict(torch.load("%s/agent%d.pth" % (model_load_path, i))) # gpu->gpu cpu->cpu
        
        actor_critic.to(device)
        actor_critics.append(actor_critic)

        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                                    args.entropy_coef, lr=args.lr,
                                    eps=args.eps, alpha=args.alpha,
                                    max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                            args.value_loss_coef, args.entropy_coef, lr=args.lr,
                            eps=args.eps,
                            max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr': # being deprecated temporarily
            agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                                    args.entropy_coef, acktr=True)
        agents.append(agent)

        rollout = RolloutStorage(args.num_pretrain_steps,
                        observation_spaces[i].shape, action_spaces[i],
                        actor_critic.recurrent_hidden_state_size, num_node)
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)
        
        
    # Pre training
    #time_costs = []
    for i in range(args.num_pretrain_epochs):
        for j in range(args.num_pretrain_steps):
            # interact with the environment
            with torch.no_grad():
                values = [None] * num_agent
                actions = [None] * num_agent
                action_log_probs = [None] * num_agent
                recurrent_hidden_states = [None] * num_agent
                condition_states = [None] * num_agent
                
                # generate routing action route by route
                curr_path = [0] * num_node
                agents_flag = [0] * num_agent
                curr_agent, path = envs.first_agent()
                
                while curr_agent != None and agents_flag[curr_agent] != 1:
                    for k in path:
                        curr_path[k] = 1
                    agents_flag[curr_agent] = 1
                    
                    # curr_path indicate current passed nodes, may be not a simple path
                    # for example agent1->node0->agent1->node0->agent2
                    condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
                    
                    #start = time.time()
                    value, action, action_log_prob, recurrent_hidden_state = actor_critics[curr_agent].act(
                            rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                            rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                            condition_state.unsqueeze(0))
                    #end = time.time()
                    #time_costs.append(end - start)

                    values[curr_agent] = value
                    actions[curr_agent] = action
                    action_log_probs[curr_agent] = action_log_prob
                    recurrent_hidden_states[curr_agent] = recurrent_hidden_state
                    condition_states[curr_agent] = condition_state
                    curr_agent, path = envs.next_agent(curr_agent, action)
                # since nodes not on the path's policy gradients will be zeroed when training, the condition state here has nothing to do with training process 
                condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
                for k in range(num_agent):
                    if agents_flag[k] != 1:
                        value, action, action_log_prob, recurrent_hidden_state = actor_critics[k].act(
                                rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                                rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                                condition_state.unsqueeze(0))
                
                        values[k] = value
                        actions[k] = action
                        action_log_probs[k] = action_log_prob
                        recurrent_hidden_states[k] = recurrent_hidden_state
                        condition_states[k] = condition_state

            # Observation reward and next obs
            gfactors = [0.] * num_agent
            obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, globalrwd, _, _, _ = envs.step(actions, gfactors, simenv=False)
            
            print(delta_dist, file=log_dist_files[rtype])
            print(delta_demand, file=log_demand_files[rtype])
            print(globalrwd, file=log_globalrwd_file)
            print(circle_flag, file=log_circle_file)
            
            for k in range(num_agent):
                # batch here must be 1 since there is only one environment
                masks = torch.tensor([1.])
                rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0), condition_states[k], actions[k].squeeze(0), action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

        for k in range(num_agent):
            # update learning rate
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                if args.algo == "acktr":
                    # use optimizer's learning rate since it's hard-coded in kfac.py
                    utils.update_linear_schedule(agents[k].optimizer, i, args.num_pretrain_epochs, agents[k].optimizer.lr)
                else:
                    utils.update_linear_schedule(agents[k].optimizer, i, args.num_pretrain_epochs, args.lr * 100)
            if args.algo == 'ppo' and args.use_linear_clip_decay:
                agents[k].clip_param = args.clip_param  * (1 - i / float(args.num_pretrain_epochs))

            # update model param
            
            with torch.no_grad(): 
                condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
                next_value = actor_critics[k].get_value(rollouts[k].obs[-1].unsqueeze(0),
                                            rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
                                            condition_state.unsqueeze(0)).detach()

                rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            value_loss, action_loss, dist_entropy = agents[k].update(rollouts[k])
    #print("avg time cost:", sum(time_costs) / len(time_costs))

    # training episode
    request, obses = envs.reset() # not reset to maintaince the stable of the network(warm start)
    # update rollouts(num-steps)
    rollouts = []
    for i in range(num_agent):
        rollout = RolloutStorage(args.num_steps,
                        observation_spaces[i].shape, action_spaces[i],
                        actor_critic.recurrent_hidden_state_size, num_node)
        rollouts.append(rollout)
        rollouts[i].obs[0].copy_(obses[i])
        rollouts[i].to(device)
    
    # update adam optimizer
    for k in range(num_agent):
        agents[k].reset_optimizer()

    # training
    for j in range(args.num_env_steps):
        # interact with the environment
        with torch.no_grad(): 
            values = [None] * num_agent
            actions = [None] * num_agent
            action_log_probs = [None] * num_agent
            recurrent_hidden_states = [None] * num_agent
            condition_states = [None] * num_agent

            # generate routing action route by route
            curr_path = [0] * num_node
            agents_flag = [0] * num_agent
            curr_agent, path = envs.first_agent()
            
            while curr_agent != None and agents_flag[curr_agent] != 1:
                for k in path:
                    curr_path[k] = 1
                agents_flag[curr_agent] = 1
                
                condition_state = torch.tensor(curr_path, dtype=torch.float32).to(device)
                value, action, action_log_prob, recurrent_hidden_state = actor_critics[curr_agent].act(
                        rollouts[curr_agent].obs[rollouts[curr_agent].step].unsqueeze(0),
                        rollouts[curr_agent].recurrent_hidden_states[rollouts[curr_agent].step].unsqueeze(0),
                        condition_state.unsqueeze(0))
                values[curr_agent] = value
                actions[curr_agent] = action
                action_log_probs[curr_agent] = action_log_prob
                recurrent_hidden_states[curr_agent] = recurrent_hidden_state
                condition_states[curr_agent] = condition_state
                curr_agent, path = envs.next_agent(curr_agent, action)
                
            # since nodes not on the path's policy gradients will be zeroed when training, the condition state here has nothing to do with training process 
            condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
            for k in range(num_agent):
                if agents_flag[k] != 1:
                    value, action, action_log_prob, recurrent_hidden_state = actor_critics[k].act(
                            rollouts[k].obs[rollouts[k].step].unsqueeze(0),
                            rollouts[k].recurrent_hidden_states[rollouts[k].step].unsqueeze(0),
                            condition_state.unsqueeze(0))
                
                    values[k] = value
                    actions[k] = action
                    action_log_probs[k] = action_log_prob
                    recurrent_hidden_states[k] = recurrent_hidden_state
                    condition_states[k] = condition_state
            
        # Observation reward and next obs
        # trans a list of gfactor may be a better choice
        gfactors = [1.] * num_agent

        obses, rewards, path, delta_dist, delta_demand, circle_flag, rtype, globalrwd, delay, throughput_rate, loss_rate = envs.step(actions, gfactors)
        print(delta_dist, file=log_dist_files[rtype])
        print(delta_demand, file=log_demand_files[rtype])
        print(delay, file=log_delay_files[rtype])
        print(throughput_rate, file=log_throughput_files[rtype])
        print(loss_rate, file=log_loss_files[rtype])
        print(globalrwd, file=log_globalrwd_file)
        print(circle_flag, file=log_circle_file)
    
        
        for k in range(num_agent):
            # batch here must be 1 since there is only one
            if agents_flag[k] == 1:
                masks = torch.tensor([1.])
            else:
                masks = torch.tensor([0.])
            rollouts[k].insert(obses[k], recurrent_hidden_states[k].squeeze(0), condition_states[k], actions[k].squeeze(0), action_log_probs[k].squeeze(0), values[k].squeeze(0), rewards[k], masks)

            # update model param
            # actually checking each agent is not essential since now the agents have same rollout steps
            if rollouts[k].step == 0:
                # update learning rate
                # asyn updating temporarily, perhaps implemented by clock or real interacting step
                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    if args.algo == "acktr":
                        # use optimizer's learning rate since it's hard-coded in kfac.py
                        utils.update_linear_schedule(agents[k].optimizer, j, args.num_env_steps, agents[k].optimizer.lr)
                    else:
                        utils.update_linear_schedule(agents[k].optimizer, j, args.num_env_steps, args.lr)
                if args.algo == 'ppo' and args.use_linear_clip_decay:
                    agents[k].clip_param = args.clip_param  * (1 - j / float(args.num_env_steps))
                
                # update actor and critic
                with torch.no_grad(): 
                    condition_state = torch.tensor([0] * num_node, dtype=torch.float32).to(device)
                    next_value = actor_critics[k].get_value(rollouts[k].obs[-1].unsqueeze(0),
                                                rollouts[k].recurrent_hidden_states[-1].unsqueeze(0),
                                                condition_state.unsqueeze(0)).detach()

                    rollouts[k].compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
                value_loss, action_loss, dist_entropy = agents[k].update(rollouts[k])

                rollouts[k].after_update()
        
        if j % ckpt_step == 0:
            if model_save_path != None:
                save_dir = os.path.expanduser(model_save_path)
                utils.cleanup_log_dir(save_dir)
                for i in range(num_agent):
                    torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))

    if model_save_path != None:
        save_dir = os.path.expanduser(model_save_path)
        utils.cleanup_log_dir(save_dir)
        for i in range(num_agent):
            torch.save(actor_critics[i].state_dict(), "%s/agent%d.pth" % (model_save_path, i))
    
    

if __name__ == "__main__":
    main()
