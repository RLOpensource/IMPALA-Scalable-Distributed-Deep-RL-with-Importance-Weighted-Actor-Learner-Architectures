import tensorflow as tf
import numpy as np

def _softmax(logits):
  """Applies softmax non-linearity on inputs."""
  return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


np.random.seed(0)
tf.set_random_seed(0)

behavior_policy = np.stack([np.random.rand(3) - 0.5 for i in range(2)])
print(behavior_policy)
action = np.stack([np.random.choice(3, p=[0.3, 0.4, 0.3]) for i in range(2)])
print(action)

print(_softmax(behavior_policy))
print(np.log(_softmax(behavior_policy)))

behavior_policy = tf.convert_to_tensor(behavior_policy)
action = tf.convert_to_tensor(action)

result = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behavior_policy, labels=action)

sess = tf.Session()
print(sess.run(result))
