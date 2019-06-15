import tensorflow as tf
import numpy as np
import threading
import impala
import config
import multiprocessing
import utils
import tensorboardX
import numpy
import gym

class Agent(threading.Thread):
    def __init__(self, session, coord, name, global_network, reward_clip, lock):
        super(Agent, self).__init__()
        self.lock = lock
        self.sess = session
        self.coord = coord
        self.name = name
        self.global_network = global_network
        self.local_network = impala.IMPALA(
            sess=self.sess,
            name=name,
            unroll=config.unroll,
            state_shape=config.state_shape,
            output_size=config.output_size,
            activation=config.activation,
            final_activation=config.final_activation,
            hidden=config.hidden,
            coef=config.entropy_coef,
            reward_clip=reward_clip
        )
        self.global_to_local = utils.copy_src_to_dst('global', name)
    
    def run(self):
        self.sess.run(self.global_to_local)
        self.env = gym.make('CartPole-v1')
        
        done = False
        state = self.env.reset()
        score = 0
        episode = 0
        episode_step = 0
        total_max_prob = 0
        loss_step = 0

        writer = tensorboardX.SummaryWriter('runs/'+self.name)

        while True:
            loss_step += 1
            episode_state = []
            episode_next_state = []
            episode_reward = []
            episode_done = []
            episode_action = []
            episode_behavior_policy = []
            for i in range(128):

                action, behavior_policy, max_prob = self.local_network.get_policy_and_action(state)
                episode_step += 1
                total_max_prob += max_prob
            
                next_state, reward, done, _ = self.env.step(action)

                score += reward

                episode_state.append(state)
                episode_next_state.append(next_state)
                episode_reward.append(reward)
                episode_done.append(done)
                episode_action.append(action)
                episode_behavior_policy.append(behavior_policy)

                state = next_state

                if done:
                    print(self.name, episode, score, total_max_prob / episode_step)
                    writer.add_scalar('score', score, episode)
                    writer.add_scalar('max_prob', total_max_prob / episode_step, episode)
                    episode_step = 0
                    total_max_prob = 0
                    episode += 1
                    score  = 0
                    done = False
                    state = self.env.reset()

            self.lock.acquire()
            pi_loss, value_loss, entropy = self.global_network.train(
                state=np.stack(episode_state),
                next_state=np.stack(episode_next_state),
                reward=np.stack(episode_reward),
                done=np.stack(episode_done),
                action=np.stack(episode_action),
                behavior_policy=np.stack(episode_behavior_policy))
            self.sess.run(self.global_to_local)
            writer.add_scalar('pi_loss', pi_loss, loss_step)
            writer.add_scalar('value_loss', value_loss, loss_step)
            writer.add_scalar('entropy', entropy, loss_step)
            self.lock.release()