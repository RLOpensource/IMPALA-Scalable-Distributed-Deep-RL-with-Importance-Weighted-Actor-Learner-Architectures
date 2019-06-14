import tensorflow as tf

def compute_value_loss(vs, value):
    error = tf.stop_gradient(vs[:, 0]) - value[:, 0]
    l2_loss = tf.square(error)
    return tf.reduce_sum(l2_loss) * 0.5

def compute_policy_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy[:, 0] * advantages[:, 0]
    return tf.reduce_sum(policy_gradient_loss_per_timestep)

def compute_entropy_loss(logits):
    policy =  tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_time_step = -policy * log_policy
    entropy_per_time_step = tf.reduce_sum(entropy_per_time_step[:, 0], axis=1)
    return -tf.reduce_sum(entropy_per_time_step)

def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return tf.reduce_sum(policy_gradient_loss_per_timestep)

def compute_baseline_loss(advantages):
    return .5 * tf.reduce_sum(tf.square(advantages))

# def compute_entropy_loss(logits):
#     policy = tf.nn.softmax(logits)
#     log_policy = tf.nn.log_softmax(logits)
#     entropy_per_time_step = tf.reduce_sum(-policy * log_policy, axis=-1)
#     return -tf.reduce_sum(entropy_per_time_step)

def log_probs_from_logits_and_actions(policy_logits, actions):
    policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions)

def from_logits(behavior_policy_logits, target_policy_logits, actions,
                discounts, rewards, values, next_value,
                clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    target_action_log_probs = log_probs_from_logits_and_actions(target_policy_logits, actions)
    behavior_action_log_probs = log_probs_from_logits_and_actions(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs

    transpose_log_rhos = tf.transpose(log_rhos, perm=[1, 0])
    transpose_discounts = tf.transpose(discounts, perm=[1, 0])
    transpose_rewards = tf.transpose(rewards, perm=[1, 0])
    transpose_values = tf.transpose(values, perm=[1, 0])
    transpose_next_value = tf.transpose(next_value, perm=[1, 0])

    transpose_vs, transpose_clipped_rho = from_importance_weights(
        log_rhos=transpose_log_rhos,
        discounts=transpose_discounts,
        rewards=transpose_rewards,
        values=transpose_values,
        bootstrap_value=transpose_next_value[-1],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    return transpose_vs, transpose_clipped_rho

def from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value,
                            clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):

    rhos = tf.exp(log_rhos)
    if clip_rho_threshold is not None:
        clipped_rhos = tf.minimum(clip_rho_threshold, rhos)
    else:
        clipped_rhos = rhos
    
    cs = tf.minimum(1.0, rhos, name='cs')
    values_t_plus_1 = tf.concat(
        [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    sequences = (discounts, cs, deltas)

    def scanfunc(acc, sequence_item):
        discount_t, c_t, delta_t = sequence_item
        return delta_t + discount_t * c_t * acc

    initial_values = tf.zeros_like(bootstrap_value)
    vs_minus_v_xs = tf.scan(
        fn=scanfunc,
        elems=sequences,
        initializer=initial_values,
        parallel_iterations=1,
        back_prop=False,
        reverse=True,
        name='scan')
    vs = tf.add(vs_minus_v_xs, values)

    return tf.stop_gradient(vs), clipped_rhos