import tensorflow as tf
import threading
import config
import impala
import multiprocessing
import async_agent

if __name__ == '__main__':
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()

    reward_clip = config.reward_clip[1]
    lock = threading.Lock()

    global_network = impala.IMPALA(
        sess=sess,
        name='global',
        unroll=config.unroll,
        state_shape=config.state_shape,
        output_size=config.output_size,
        activation=config.activation,
        final_activation=config.final_activation,
        hidden=config.hidden,
        coef=config.entropy_coef,
        reward_clip=reward_clip
    )

    n_threads = 16

    thread_list = []

    for i in range(n_threads):
        single_agent = async_agent.Agent(
            session=sess,
            coord=coord,
            name='thread_{}'.format(i),
            global_network=global_network,
            reward_clip=reward_clip,
            lock=lock
        )

        thread_list.append(single_agent)

    init = tf.global_variables_initializer()
    sess.run(init)

    for t in thread_list:
        t.start()
