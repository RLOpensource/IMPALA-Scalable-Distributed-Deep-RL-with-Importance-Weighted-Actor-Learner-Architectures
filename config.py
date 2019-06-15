import tensorflow as tf

state_shape = [80, 80, 4]
batch_size = 32
output_size = 2
activation = tf.nn.relu
final_activation = None
hidden = [256, 256, 256]
unroll = 20
entropy_coef = 0.00025
reward_clip = ['tanh', 'abs_one', 'no_clip']

learner_ip = '0.0.0.0'
send_size = 32