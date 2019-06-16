# Implementation of Scalable-Distributed-Deep-RL-with-Importance-Weighted-Actor-Learner-Architectures

* These results are from only 4 threads. So unstable to train.
* Tensorflow Implementation
* A3C type thread environment training method
* PongDeterministic-v4 environment

<div align="center">
  <img src="source/video.gif" width="50%" height='300'>
</div>

<div align="center">
  <img src="source/entropy.png" width="32%" height='300'>
  <img src="source/episode_step.png" width="33%" height='300'>
  <img src="source/max_prob.png" width="33%" height='300'>
  <img src="source/pi_loss.png" width="32%" height='300'>
  <img src="source/score.png" width="33%" height='300'>
  <img src="source/value_loss.png" width="33%" height='300'>
</div>

# Todo

- [x] Only CPU Training method
- [ ] Use Network protocol method
- [ ] Training on GPU, Inference on CPU

# Reference

* [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
* [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
* [Asynchronous_Advatnage_Actor_Critic](https://github.com/RLOpensource/Asynchronous_Advatnage_Actor_Critic)