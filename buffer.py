import collections
import numpy as np
import random

class unroll_buffer:
    def __init__(self, unroll=10):
        self.unroll = unroll
        self.length = 0
        self.state = collections.deque(maxlen=self.unroll)
        self.next_state = collections.deque(maxlen=self.unroll)
        self.action = collections.deque(maxlen=self.unroll)
        self.reward = collections.deque(maxlen=self.unroll)
        self.done = collections.deque(maxlen=self.unroll)
        self.behavior_policy = collections.deque(maxlen=self.unroll)

    def reset(self):
        self.length = 0
        self.state = collections.deque(maxlen=self.unroll)
        self.next_state = collections.deque(maxlen=self.unroll)
        self.action = collections.deque(maxlen=self.unroll)
        self.reward = collections.deque(maxlen=self.unroll)
        self.done = collections.deque(maxlen=self.unroll)
        self.behavior_policy = collections.deque(maxlen=self.unroll)

    def append(self, state, next_state, action, reward, done, behavior_policy):
        self.state.append(state)
        self.next_state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.behavior_policy.append(behavior_policy)
        self.length += 1
        if self.length > self.unroll:
            self.length = self.unroll

    def sample(self):
        if self.length == self.unroll:
            return np.stack(self.state), np.stack(self.next_state), np.stack(self.action), np.stack(self.reward), np.stack(self.done), np.stack(self.behavior_policy)
        else:
            return None

class replay_buffer:
    def __init__(self, max_length=1e6):
        self.max_length = max_length
        self.key = ['state', 'next_state', 'reward', 'done', 'action', 'behavior_policy']
        self.memory = collections.deque(maxlen=int(self.max_length))

    def reset(self):
        self.memory = collections.deque(maxlen=int(self.max_length))

    def append(self, state, next_state, reward, done, action, behavior_policy):
        self.memory.append((state, next_state, reward, done, action, behavior_policy))

    def get_sample(self, sample_size):
        batch = random.sample(self.memory, sample_size)
        state = np.stack([e[0] for e in batch])
        next_state = np.stack([e[1] for e in batch])
        reward = np.stack([e[2] for e in batch])
        done = np.stack([e[3] for e in batch])
        action = np.stack([e[4] for e in batch])
        behavior_policy = np.stack([e[5] for e in batch])
        batch_memory = [state, next_state, reward, done, action, behavior_policy]

        return {k:v for k, v in zip(self.key, batch_memory)}

    def get_train_data(self):
        batch = random.sample(self.memory, 1)
        state = np.stack([e[0] for e in batch])
        next_state = np.stack([e[1] for e in batch])
        reward = np.stack([e[2] for e in batch])
        done = np.stack([e[3] for e in batch])
        action = np.stack([e[4] for e in batch])
        behavior_policy = np.stack([e[5] for e in batch])
        batch_memory = [state, next_state, reward, done, action, behavior_policy]

        return {k:v for k, v in zip(self.key, batch_memory)}
 
class impala_buffer:
    def __init__(self, capacity=int(1e5)):
        self.state = []
        self.next_state = []
        self.action = []
        self.reward = []
        self.done = []
        self.behavior_policy = []
        self.capacity = capacity

    def append(self, state, next_state, reward, done, action, behavior_policy):
        if len(self.state) > self.capacity:
            self.state = self.state[1:]
            self.next_state = self.next_state[1:]
            self.reward = self.reward[1:]
            self.done = self.done[1:]
            self.action = self.action[1:]
            self.behavior_policy = self.behavior_policy[1:]

        self.state.append(state)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.done.append(done)
        self.action.append(action)
        self.behavior_policy.append(behavior_policy)

    def sample(self, sample_size, ratio):
        if sample_size > len(self.state):
            return None
        sample_idx = np.arange(len(self.state))
        np.random.shuffle(sample_idx)
        shuffled_idx = sample_idx[:sample_size]
        shuffled_idx = sorted(shuffled_idx)
        shuffled_idx.reverse()

        sampled_state = []
        sampled_next_state = []
        sampled_reward = []
        sampled_done = []
        sampled_action = []
        sampled_behavior_policy = []

        for i in shuffled_idx:
            sampled_state.append(self.state.pop(i))
            sampled_next_state.append(self.next_state.pop(i))
            sampled_reward.append(self.reward.pop(i))
            sampled_done.append(self.done.pop(i))
            sampled_action.append(self.action.pop(i))
            sampled_behavior_policy.append(self.behavior_policy.pop(i))

        append_idx = np.arange(sample_size)
        np.random.shuffle(append_idx)
        append_idx = append_idx[:int(sample_size * ratio)]

        for i in append_idx:
            self.state.append(sampled_state[i])
            self.next_state.append(sampled_next_state[i])
            self.reward.append(sampled_reward[i])
            self.done.append(sampled_done[i])
            self.action.append(sampled_action[i])
            self.behavior_policy.append(sampled_behavior_policy[i])

        return np.stack(sampled_state), np.stack(sampled_next_state), np.stack(sampled_reward), np.stack(sampled_done), np.stack(sampled_action), np.stack(sampled_behavior_policy)

class temp_buffer:
    def __init__(self, capacity=int(1e5)):
        self.state = collections.deque(maxlen=capacity)
        self.next_state = collections.deque(maxlen=capacity)
        self.action = collections.deque(maxlen=capacity)
        self.reward = collections.deque(maxlen=capacity)
        self.done = collections.deque(maxlen=capacity)
        self.behavior_policy = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def append(self, state, next_state, action, reward, done, behavior_policy):
        self.state.append(state)
        self.next_state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.behavior_policy.append(behavior_policy)

    def clear(self):
        self.state = collections.deque(maxlen=self.capacity)
        self.next_state = collections.deque(maxlen=self.capacity)
        self.action = collections.deque(maxlen=self.capacity)
        self.reward = collections.deque(maxlen=self.capacity)
        self.done = collections.deque(maxlen=self.capacity)
        self.behavior_policy = collections.deque(maxlen=self.capacity)

    def sample(self):
        if len(self.state) != self.capacity:
            return None
        else:
            return np.stack(self.state), np.stack(self.next_state), np.stack(self.action), np.stack(self.reward), np.stack(self.done), np.stack(self.behavior_policy)