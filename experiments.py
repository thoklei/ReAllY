import gym
import numpy as np
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    pass
import time
import ray
import os
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)
import ppo_model
"""
#####    SUMMARY     #####

----- Initialization -----
    Setup - some constants are set.
    Experiment settings - the settings we tweaked during our experiments.
    Hyperparameter - here are all parameters for training and all models set.
    IO - here are common parameters initialized for IO transactions throughout the training phase.

=====    Training    =====
    Setup - Involves sampling and data enhancement.
    Train - Sampled data is used to train the policy.
    Testing - Improved policy gets assessed.
    IO - Predefined IO operations are performed.
    
+++++      End:      +++++ 
    x - Final performance of the policy is shown.
"""

if __name__ == "__main__":
    # ----------------------        Setup        ----------------------
    tf.random.set_seed(42)

    bipedal = "BipedalWalker-v3"
    lunar = "LunarLanderContinuous-v2"

    for e_name in [bipedal, lunar]:
        if not os.path.exists("./" + e_name):
            os.makedirs(e_name)

    # ---------------------- Experiment settings ----------------------
    """ For our experiments we only changed those parameters. """
    env_name = lunar  # choose env name, either bipedal or lunar
    use_rnd = True  # whether to use vanilla PPO or PPO + RND
    continue_from_saved_model = False
    use_ray = True

    if use_ray:
        # To disable ray successfully comment out '@ray.remote' in really.runner_box.py
        # This can increase computation time significantly when not sampling too long trajectories.
        ray.init(log_to_driver=False)

    # ----------------------    Hyperparameter   ----------------------
    """ The following parameters were tuned once and then kept through out the experiments."""
    learning_rate = 0.001
    max_episodes = 300
    test_steps = 1600
    sampled_batches = 512
    optimization_batch_size = 64
    gamma = 0.99
    my_lambda = 0.95
    clipping_value = 0.3
    critic_discount = 0.5
    entropy_beta = 0.001

    # Model kwargs
    with gym.make(env_name) as env:
        action_dim = env.action_space.shape[0]
        state_dim = env.observation_space.shape[0]
        solving_threshold = env.spec.reward_threshold
    middle_layers = [48, 48, 48] if env_name == bipedal else [32, 32, 32]
    model_kwargs = {"layers": middle_layers, "action_dim": action_dim, "state_dim": state_dim}

    ppo_loss = tf.keras.losses.MeanSquaredError()
    ppo_optimizer = tf.keras.optimizers.Adam(learning_rate)

    if use_rnd:
        curiosity_weight = 0.5
        curiosity_gamma = 0.98

        layers = [8, 8]
        k = 8
        # fixed random net used to obtain features
        target_network = ppo_model.TargetNetwork(layers, k, state_dim)
        target_network.trainable = False
        # predictor network we will train to match target network
        predictor = ppo_model.TargetNetwork(layers, k, state_dim)

        pred_optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Instantiate and Initialize Sample Manager
    manager = SampleManager(
        model=ppo_model.A2C,
        environment=env_name,
        num_parallel=3,
        total_steps=420,
        returns=['value_estimate', 'log_prob', 'monte_carlo'],
        model_kwargs=model_kwargs,
        action_sampling_type="continuous_normal_diagonal",
        use_ray=use_ray
    )

    # ----------------------          IO         ----------------------
    """
    This section saves plots of the training process, writes some details to csv, 
    and allow to continue training from an existing models.
    """
    saving_path = os.getcwd() + "/" + env_name
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", 'reward', 'time']
    )

    results_file_name = env_name + f"/results_{'rnd' if use_rnd else 'base'}.csv"
    with open(results_file_name, 'a') as fd:
        fd.write('epoch,loss,reward,rnd_loss,steps\n')

    if continue_from_saved_model:
        agent, epoch_offset = manager.load_model(saving_path)
    else:
        agent = manager.get_agent()
        epoch_offset = 0

    # ======================      Training       ======================
    rewards = []
    print('TRAINING')
    for e in range(epoch_offset, max_episodes + epoch_offset):
        e += 1
        t = time.time()

        # ======================     Setup       ======================
        """ 
        In the Setup phase we will:
        1. sample trajectories
        2. compute advantages 
            2a. compute the GAE
            2b. include an intrinsic reward to the GAE signal.
            2c. Zero-center the advantages.
        """
        # 1. Sample data to optimize
        sample_dict = manager.sample(
            sample_size=sampled_batches * optimization_batch_size,
            from_buffer=False
        )

        # 2. Compute Advantages
        # Add value estimate of last 'new_state'
        sample_dict['value_estimate'].append(agent.v(np.expand_dims(sample_dict['state_new'][-1], 0)))
        sample_dict['advantage'] = []
        gae = 0
        intrinsic = 0
        # Loop backwards through rewards
        for i in reversed(range(len(sample_dict['reward']))):
            # 2a.
            delta = sample_dict['reward'][i] + gamma * sample_dict['value_estimate'][i + 1].numpy() * \
                    sample_dict['not_done'][i] - sample_dict['value_estimate'][i].numpy()

            gae = delta + gamma * my_lambda * sample_dict['not_done'][i] * gae

            if use_rnd:
                # 2b.
                intrinsic = tf.squeeze(
                    tf.keras.losses.MSE(
                        target_network(sample_dict['state'][i].reshape(1, -1)),
                        predictor(sample_dict['state'][i].reshape(1, -1))
                    )
                ).numpy() + sample_dict['not_done'][i] * curiosity_gamma * intrinsic
                gae = gae + curiosity_weight * intrinsic

            # Insert advantage in front to keep correct order
            sample_dict['advantage'].insert(0, gae)

        # 2c. Center advantages around zero
        sample_dict['advantage'] -= np.mean(sample_dict['advantage'])

        # Remove keys that are no longer used to free up memory
        [sample_dict.pop(key) for key in ["value_estimate", "state_new", "reward", "not_done"]]

        # Convert the sample dict to a dict of tf datasets.
        samples = dict_to_dict_of_datasets(sample_dict, batch_size=optimization_batch_size)
        del sample_dict  # free up memory

        # ======================     Train      ======================
        """
        The ppo agent gets improved on the enhanced, sampled data,
        as well as the predictor network if RND is used.
        """

        actor_losses, critic_losses, losses, advantages, rnd_losses = [], [], [], [], []
        for state_batch, action_batch, advantage_batch, returns_batch, log_prob_batch in zip(
                samples['state'], samples['action'], samples['advantage'],
                samples['monte_carlo'], samples['log_prob']
        ):
            if use_rnd:
                rnd_loss = ppo_model.train_on_batch_rnd(
                    state_batch,  # Data
                    target_network, predictor,  # Networks
                    pred_optimizer  # optimizer for predictor
                )
                rnd_losses.append(rnd_loss)

            actor_loss, critic_loss, total_loss = ppo_model.train_on_batch_ppo(
                agent,
                # Data
                state_batch, action_batch, advantage_batch, log_prob_batch, returns_batch,
                # Hyperparameters
                clipping_value, critic_discount, entropy_beta,
                # Loss and Optimizer
                ppo_loss, ppo_optimizer
            )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            losses.append(total_loss)
            advantages.append(tf.reduce_mean(advantage_batch))

        manager.set_agent(agent.get_weights())

        # ======================     Testing      ======================

        # Update aggregator
        steps, current_rewards = manager.test(
            max_steps=test_steps,
            test_episodes=5,
            render=False,
            evaluation_measure="time_and_reward"
        )

        # Collect all rewards
        rewards.extend(current_rewards)
        # Average reward over last 100 episodes
        avg_reward = sum(rewards[-100:]) / min(len(rewards), 100)

        # Print progress
        print(
            f"e: {e}   loss: {np.mean(losses):.3f}   RND loss: {(curiosity_weight * np.mean(rnd_losses)) if use_rnd else 0:.6f}"
            f"   adv: {np.mean(advantages):.6f}   avg_curr_rew: {np.mean(current_rewards):.3f}   "
            f"avg_reward: {avg_reward:.3f}   avg_steps: {np.mean(steps):.2f}   time {(time.time() - t):.2f}"
        )

        # ======================        IO        ======================

        manager.update_aggregator(loss=losses, reward=current_rewards, time=steps)

        # print progress to file
        with open(results_file_name, 'a') as fd:
            fd.write(','.join(
                [str(np.mean(x)) for x in [e, losses, current_rewards,
                                           (curiosity_weight * np.mean(rnd_losses)) if use_rnd else 0, steps]]
            ) + '\n')

        if (e % 25) == 0:
            manager.save_model(saving_path, e, model_name='model')

        if avg_reward >= solving_threshold:
            # if the env is solved the model will be saved and the training is completed
            manager.save_model(saving_path, e, model_name='solved')
            break

    # ++++++++++++++++++++++       End       ++++++++++++++++++++++
    print("\nTesting optimized agent")
    manager.test(
        max_steps=1000,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
