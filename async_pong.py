import tensorflow as tf
import numpy as np
import impala
import config_pong
import multiprocessing
import utils
import tensorboardX
import gym
import buffer
import copy
import cv2
import threading

class Agent(threading.Thread):
    def __init__(self, session, coord, name, global_network, reward_clip):
        super(Agent, self).__init__()
        self.sess = session
        self.name = name
        self.coord = coord
        self.global_network = global_network
        self.local_network = impala.IMPALA(
            sess=self.sess,
            name=name,
            unroll=config_pong.unroll,
            state_shape=config_pong.state_shape,
            output_size=config_pong.output_size,
            activation=config_pong.activation,
            final_activation=config_pong.final_activation,
            hidden=config_pong.hidden,
            coef=config_pong.entropy_coef,
            reward_clip=reward_clip
        )
        self.global_to_local = utils.copy_src_to_dst('global', name)

    def run(self):
        self.sess.run(self.global_to_local)
        self.env = gym.make('PongDeterministic-v4')
        self.seed = np.random.randint(100)
        print('seed :', self.name, self.seed)
        self.env.seed(self.seed)

        temp_buffer = buffer.temp_buffer(capacity=int(config_pong.unroll))
        impala_buffer = buffer.impala_buffer(capacity=int(config_pong.send_size))

        done = False
        _ = self.env.reset()
        frame = self.pre_proc(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
        history = np.stack((frame, frame, frame, frame), axis=2)
        state = copy.deepcopy(history)
        score = 0
        episode = 0
        episode_step = 0
        total_max_prob = 0
        loss_step = 0

        writer = tensorboardX.SummaryWriter('runs/'+self.name)

        while True:
            action, behavior_policy, max_prob = self.local_network.get_policy_and_action(state)
            episode_step += 1
            total_max_prob += max_prob

            _, reward, done, _ = self.env.step(action+1)
            frame = self.pre_proc(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
            history[:, :, :3] = history[:, :, 1:]
            history[:, :, 3] = frame
            next_state = copy.deepcopy(history)

            score += reward

            temp_buffer.append(
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                done=done,
                behavior_policy=behavior_policy
            )
            trajectory = temp_buffer.sample()
            if trajectory is not None:
                impala_buffer.append(
                    state=trajectory[0],
                    next_state=trajectory[1],
                    action=trajectory[2],
                    reward=trajectory[3],
                    done=trajectory[4],
                    behavior_policy=trajectory[5]
                )

            train_data = impala_buffer.sample(config_pong.send_size, 0)
            if train_data is not None:
                loss_step += 1
                train_length = np.arange(config_pong.send_size)
                np.random.shuffle(train_length)
                train_idx = train_length[:config_pong.batch_size]
                pi_loss, value_loss, entropy, gradient = self.global_network.train(
                    state=[train_data[0][i] for i in train_idx],
                    next_state=[train_data[1][i] for i in train_idx],
                    reward=[train_data[2][i] for i in train_idx],
                    done=[train_data[3][i] for i in train_idx],
                    action=[train_data[4][i] for i in train_idx],
                    behavior_policy=[train_data[5][i] for i in train_idx]
                )
                self.sess.run(self.global_to_local)
                writer.add_scalar('pi_loss', pi_loss, loss_step)
                writer.add_scalar('value_loss', value_loss, loss_step)
                writer.add_scalar('entropy', entropy, loss_step)

            state = next_state

            if done:
                writer.add_scalar('episode_step', episode_step, episode)
                writer.add_scalar('score', score, episode)
                writer.add_scalar('max_prob', total_max_prob / episode_step, episode)
                print(self.name, episode, score, total_max_prob / episode_step)
                done = False
                _ = self.env.reset()
                frame = self.pre_proc(self.env.env.ale.getScreenGrayscale().squeeze().astype('float32'))
                history = np.stack((frame, frame, frame, frame), axis=2)
                state = copy.deepcopy(history)
                score = 0
                episode += 1
                episode_step = 0
                total_max_prob = 0

    def pre_proc(self, x):
        x = cv2.resize(x, (84, 84))
        x *= (1.0 / 255.0)
        return x

    
if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.Session()
    coord = tf.train.Coordinator()

    reward_clip = config_pong.reward_clip[1]

    global_network = impala.IMPALA(
        sess=sess,
        name='global',
        unroll=config_pong.unroll,
        state_shape=config_pong.state_shape,
        output_size=config_pong.output_size,
        activation=config_pong.activation,
        final_activation=config_pong.final_activation,
        hidden=config_pong.hidden,
        coef=config_pong.entropy_coef,
        reward_clip=reward_clip
    )

    n_threads = multiprocessing.cpu_count()

    thread_list = []
    for i in range(n_threads):
        single_agent = Agent(
            session=sess,
            coord=coord,
            name='thread_{}'.format(i),
            global_network=global_network,
            reward_clip=reward_clip
        )

        thread_list.append(single_agent)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    for t in thread_list:
        t.start()
