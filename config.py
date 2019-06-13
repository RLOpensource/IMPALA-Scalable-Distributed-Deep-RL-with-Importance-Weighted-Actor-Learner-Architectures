import tensorflow as tf

state_shape = [4]
batch_size = 32
output_size = 2
activation = tf.nn.relu
final_activation = None
hidden = [512, 512, 512]
unroll = 5
entropy_coef = 0.00025
reward_clip = ['tanh', 'abs_one', 'no_clip']

learner_ip = '0.0.0.0'
send_size = 32