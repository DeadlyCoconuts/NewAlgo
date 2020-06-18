import gym
import numpy as np
import matplotlib.pyplot as plt


def get_state_space(env):
    size_state_space_vector = env.observation_space.high - env.observation_space.low + 1
    num_states = np.prod(size_state_space_vector)
    return num_states


def get_action_space(env):
    size_action_space = env.action_space.n
    return size_action_space


def get_idx_from_tuple(tuple, size):
    x_value = tuple[0]
    y_value = tuple[1]
    idx = y_value * size[1] + x_value
    return idx

def plot_state_distribution(ref_dict, act_dict, time_steps, size, num_rewards, reward_type):
    x_list = [str(i) for i in range(0, len(ref_dict))]  # array to represent x axis

    plt.figure(figsize=(10, 8))  # x, x-2
    plt.bar(x_list, ref_dict, width=1, color='b', label='Benchmark Distribution')
    plt.bar(x_list, act_dict, width=1, color='r', label='Actual Distribution', alpha=0.5)
    plt.legend()
    plt.xlabel('State Number')
    plt.ylabel('Percentage of Total Time Steps Spent in State')
    plt.title('State Density Distribution for a {} Maze with {} Goals for $T = {}$ Time Steps (SSE)'.format(size, num_rewards, time_steps))

    plt.savefig('./graphics/test{}-{}-{}-distribution.png'.format(size, time_steps, reward_type))

def plot_regret_evolution(res_dict, benchmark_optimum, time_steps, size, num_rewards):
    x_list = range(time_steps)

    y_list = benchmark_optimum - res_dict

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(x_list, y_list)
    plt.xlabel('Time Steps $T$')
    plt.ylabel('Regret $Reg(T)$')
    plt.title('Evolution of $Reg(T)$ over Time $T$ for a {} Maze with {} Goals for $T = {}$ Time Steps (KPI)'.format(size, num_rewards, time_steps))
    ax.set_yscale('log')
    ax.set_xscale('log')

    plt.savefig('./graphics/test{}-{}-{}.png'.format(size, time_steps))

def plot_target_evolution(target, v_goal, time_steps, num_rewards):
    x_list = range(time_steps)

    #num_subplots = len(v_goal.keys())

    fig, axs = plt.subplots(num_rewards, figsize=(10, 10))

    for i in range(num_rewards):
        axs[i].plot(x_list, [target[i]] * time_steps, '--', color='r', label='Target')
        #axs[i].plot(x_list, [benchmark_optimum[i]] * time_steps, ':', color='b', label='Benchmark Optimum $P_M$')
        axs[i].plot(x_list, v_goal[i], color='g', label='Algorithm Optimum')
        axs[i].set_title('Objective {}'.format(i))
        axs[i].set_xscale('log')
        axs[i].legend()

    fig.suptitle('Evolution of Multiobjective Values over Time $T$ with {} Goals for '
                 '$T = {}$ Time Steps'.format(num_rewards, time_steps))

    plt.savefig('./graphics/test{}-target.png'.format(time_steps))

