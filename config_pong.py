import tensorflow as tf

state_shape = [84, 84, 4]
batch_size = 32
output_size = 3
activation = tf.nn.relu
final_activation = None
hidden = [256, 256]
unroll = 40
entropy_coef = 0.000025
reward_clip = ['tanh', 'abs_one', 'no_clip']

learner_ip = '0.0.0.0'
send_size = 32