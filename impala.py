import core
import tensorflow as tf
import numpy as np
import config
import vtrace

class IMPALA:
    def __init__(self, sess, name, unroll, state_shape, output_size, activation, final_activation, hidden, coef, reward_clip):
        self.sess = sess
        self.state_shape = state_shape
        self.output_size = output_size
        self.activation = activation
        self.final_activation = final_activation
        self.hidden = hidden
        self.clip_rho_threshold = 1.0
        self.clip_pg_rho_threshold = 1.0
        self.discount_factor = 0.99
        self.lr = 0.00048
        self.unroll = unroll
        self.coef = coef
        self.reward_clip = reward_clip

        self.s_ph = tf.placeholder(tf.float32, shape=[None, self.unroll, *self.state_shape])
        self.a_ph = tf.placeholder(tf.int32, shape=[None, self.unroll])
        self.d_ph = tf.placeholder(tf.bool, shape=[None, self.unroll])
        self.behavior_policy = tf.placeholder(tf.float32, shape=[None, self.unroll, self.output_size])
        self.r_ph = tf.placeholder(tf.float32, shape=[None, self.unroll])

        if self.reward_clip == 'tanh':
            squeezed = tf.tanh(self.r_ph / 5.0)
            self.clipped_rewards = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.
        elif self.reward_clip == 'abs_one':
            self.clipped_rewards = tf.clip_by_value(self.r_ph, -1.0, 1.0)
        elif self.reward_clip == 'no_clip':
            self.clipped_rewards = self.r_ph

        self.discounts = tf.to_float(~self.d_ph) * self.discount_factor

        self.policy, self.value = core.build_model(
            self.s_ph, self.hidden, self.activation, self.output_size,
            self.final_activation, self.state_shape, self.unroll, name
        )

        self.policy_probability = tf.nn.softmax(self.policy)

        self.transpose_vs, self.transpose_pg_advantage = vtrace.from_logits(self.behavior_policy, self.policy, self.a_ph,
                                         self.discounts, self.clipped_rewards, self.value)

        self.vs = tf.transpose(self.transpose_vs, perm=[1, 0])
        self.pg_advantage = tf.transpose(self.transpose_pg_advantage, perm=[1, 0])
        
        self.pi_loss = vtrace.compute_policy_gradient_loss(self.policy, self.a_ph, self.pg_advantage)
        self.value_loss = 0.5 * vtrace.compute_baseline_loss(self.vs - self.value)
        self.entropy = vtrace.compute_entropy_loss(self.policy)

        self.total_loss = self.pi_loss + self.value_loss + self.entropy * self.coef

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, epsilon=0.1, momentum=0.0, decay=0.99)
        self.gradients, self.gradient_variable = zip(*self.optimizer.compute_gradients(self.total_loss))
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 40.0)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.gradient_variable))

    def train(self, state, next_state, reward, done, action, behavior_policy):
        feed={
            self.s_ph: state,
            self.r_ph: reward,
            self.d_ph: done,
            self.a_ph: action,
            self.behavior_policy: behavior_policy
        }
        
        policy_loss, value_loss, entropy, gradient, _ = self.sess.run(
            [self.pi_loss, self.value_loss, self.entropy, self.gradients, self.train_op],
            feed_dict=feed
        )
        
        return policy_loss, value_loss, entropy, gradient

    def variable_to_network(self, variable):
        feed_dict={i:j for i, j in zip(self.from_list, variable)}
        self.sess.run(self.write_main_parameter, feed_dict=feed_dict)

    def get_parameter(self):
        variable = self.sess.run(self.variable)
        return variable

    def get_policy_and_action(self, state):
        state = [state for i in range(self.unroll)]
        policy, logits = self.sess.run([self.policy_probability, self.policy], feed_dict={self.s_ph: [state]})
        policy = policy[0][0]
        logits = logits[0][0]
        action = np.random.choice(self.output_size, p=policy)
        return action, logits, max(policy)

    def test(self, state, action, reward, done, behavior_policy):
        feed_dict={
            self.s_ph: state,
            self.a_ph: action,
            self.d_ph: done,
            self.behavior_policy: behavior_policy,
            self.r_ph: reward
        }


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    episode_length = config.send_size
    batch_size = config.batch_size
    unroll = config.unroll

    state_shape = config.state_shape
    output_size = config.output_size

    state = np.random.rand(episode_length, *state_shape)
    next_state = [state[i+1] for i in range(episode_length - 1)]
    next_state.append(np.random.rand(*state_shape))
    next_state = np.stack(next_state)

    done = []
    for i in range(episode_length):
        if ((i+1) % 20) == 0:
            done.append(True)
        else:
            done.append(False)
    
    reward = [np.random.choice(3, p=[0.2, 0.7, 0.1])-1 for i in range(episode_length)]
    action = [np.random.choice(output_size, p=[0.3, 0.4, 0.3]) for i in range(episode_length)]
    behavior_policy = [np.random.rand(output_size)-0.5 for i in range(episode_length)]
    
    start_index = np.arange(episode_length-unroll)
    np.random.shuffle(start_index)
    sample_start_index = np.stack(start_index[:batch_size])
    sample_end_index = np.stack(sample_start_index) + unroll

    sampled_state = [state[s:e] for s, e in zip(sample_start_index, sample_end_index)]
    sampled_next_state = [next_state[s:e] for s, e in zip(sample_start_index, sample_end_index)]
    sampled_reward = [reward[s:e] for s, e in zip(sample_start_index, sample_end_index)]
    sampled_action = [action[s:e] for s, e in zip(sample_start_index, sample_end_index)]
    sampled_behavior_policy = [behavior_policy[s:e] for s, e in zip(sample_start_index, sample_end_index)]
    sampled_done = [done[s:e] for s, e in zip(sample_start_index, sample_end_index)]

    ## sampled data shape
    ## sampled_state = [batch_size, unroll, *state_shape]
    ## sampled_next_state = [batch_size, unroll, *state_shape]
    ## sampled_reward = [batch_size, unroll]
    ## sampled_done = [batch_size, unroll]
    ## sampled_action = [batch_size, unroll]
    ## sampled_behavior_policy = [batch_size, unroll, output_size]

    sess = tf.Session()

    agent = IMPALA(
        sess=sess,
        name='global',
        unroll=config.unroll,
        state_shape=state_shape,
        output_size=output_size,
        activation=config.activation,
        final_activation=config.final_activation,
        hidden=config.hidden,
        coef=config.entropy_coef,
        reward_clip=config.reward_clip[1]
    )

    init = tf.global_variables_initializer()
    sess.run(init)
    
    agent.test(
        state=sampled_state,
        action=sampled_action,
        reward=sampled_reward,
        done=sampled_done,
        behavior_policy=sampled_behavior_policy
    )