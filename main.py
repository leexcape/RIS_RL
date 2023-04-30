from importlib import reload
import math
import matplotlib.pyplot as plt
import DQN_model
import argparse
import torch
import numpy as np
from itertools import count
import matlab.engine
import environment

reload(DQN_model)
reload(environment)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="RIS")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--max_length_of_trajectory', default=20, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)


# eng = matlab.engine.connect_matlab()
# # eng.cd('D:/SynologyDrive/coding project/MATLAB/RIS D2D')
# eng.warning('off', nargout=0)
# env = eng.RIS_ENV

state_dim = 20
action_dim = 64
max_action = math.pi
min_action = -math.pi

agent = DQN_model.DDPG(state_dim, action_dim, max_action, args.capacity)
ep_r = 0
if args.mode == 'test':
    # agent.load()
    for i in range(args.test_iteration):
        # _, state = env(np.random.rand(action_dim), nargout=2)
        # state = np.array(state[0])
        large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, aoa, aod = environment.get_channel()
        state = np.concatenate((aoa, aod), 0)[:, 0]
        for t in count():
            action = agent.select_action(state)
            reward = environment.execute_channel(large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, action)
            large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, aoa, aod = environment.get_channel()
            next_state = np.concatenate((aoa, aod), 0)[:, 0]
            # reward, next_state = env(np.float32(action), nargout=2)
            # next_state = np.array(next_state[0])
            ep_r += reward
            if t >= args.max_length_of_trajectory:
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                ep_r = 0
                break
            state = next_state

elif args.mode == 'train':
    if args.load: agent.load()
    total_step = 0
    acc_reward_rec = []
    for i in range(args.max_episode):
        total_reward = 0
        step = 0
        large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, aoa, aod = environment.get_channel()
        state = np.concatenate((aoa, aod), 0)[:, 0]
        for t in range(20):
            action = agent.select_action(state)
            action = (action + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(
                min_action, max_action)
            reward = environment.execute_channel(large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, action)
            large_scale_fading_a, large_scale_fading_b, p_t, hai, hbi, aoa, aod = environment.get_channel()
            next_state = np.concatenate((aoa, aod), 0)[:, 0]
            # reward, next_state = env(np.float32(action), nargout=2)
            # next_state = np.array(next_state[0])[0, :]
            # print('current reward: ', reward)
            # if reward < 800:
            #     done = True
            #     break
            # else:
            #     done = False
            done = False
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))

            state = next_state
            step += 1
            total_reward += reward
        total_step += step+1
        acc_reward_rec.append(total_reward)
        print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
        if len(agent.replay_buffer.storage) > args.batch_size:
            agent.update(update_iteration=args.update_iteration, batch_size=args.batch_size, gamma=args.gamma, tau=args.tau)

        if i % args.log_interval == 0:
            agent.save()

    plt.plot(acc_reward_rec)
    plt.show()
else:
    raise NameError("mode wrong!!!")

