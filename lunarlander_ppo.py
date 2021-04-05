import gym
import numpy as np
import tensorflow as tf
import ray
import os
from really import SampleManager
from really.utils import (
    dict_to_dict_of_datasets,
)
from really.utils import discount_cumsum
from tensorflow.keras import Model

#from ppo_model import ModelWrapper
from sample_model import A2C
#from improved_ppo import A2C

tf.random.set_seed(42)

if __name__ == "__main__":
    
    env = gym.make("LunarLanderContinuous-v2")
    env.seed(42)

    model_kwargs = {"layers": [32,32,32], "action_dim": env.action_space.shape[0]}
    #model_kwargs = {"state_size": 8, "batch_size": 64}
    
    learning_rate = 0.001
    max_episodes = 300
    sampled_batches = 512
    optimization_batch_size= 64
    gamma = 0.99
    my_lambda = 0.95
    clipping_value = 0.3   
    critic_discount = 0.5
    entropy_beta = 0.001

    kwargs = {
        "model": A2C,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 3,
        "total_steps": 420,
        "returns": ['value_estimate', 'log_prob', 'monte_carlo'],
        "model_kwargs": model_kwargs,
        "action_sampling_type": "continuous_normal_diagonal",
        #"gamma": gamma
    }

    # Initialize the loss function
    mse_loss = tf.keras.losses.MeanSquaredError()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initilize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    # Where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_test"#"/progress_LunarLanderContinuous"

    # Initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", 'reward', 'time']
    )

    rewards = []

    # Get initial agent
    agent = manager.get_agent()

    print('TRAINING')
    for e in range(max_episodes):
        
        # Sample data to optimize
        #print('sampling...')
        sample_dict = manager.sample(
            sample_size = sampled_batches*optimization_batch_size,
            from_buffer = False
            )
        
        # Compute Advantages
        #print('calculate advantage estimates...')

        # Add value of last 'new_state'
        sample_dict['value_estimate'].append(agent.v(np.expand_dims(sample_dict['state_new'][-1],0)))

        sample_dict['advantage'] = []
        gae = 0
        # Loop backwards through rewards
        for i in reversed(range(len(sample_dict['reward']))):
            delta = sample_dict['reward'][i] + gamma * sample_dict['value_estimate'][i+1].numpy() * sample_dict['not_done'][i] - sample_dict['value_estimate'][i].numpy()
            gae = delta + gamma * my_lambda * sample_dict['not_done'][i] * gae
            # Insert advantage in front to get correct order
            sample_dict['advantage'].insert(0, gae)
        # Center advantage around zero
        sample_dict['advantage'] -= np.mean(sample_dict['advantage'])

        # Remove keys that are no longer used
        sample_dict.pop('value_estimate')
        sample_dict.pop('state_new')
        sample_dict.pop('reward')
        sample_dict.pop('not_done')

        samples = dict_to_dict_of_datasets(sample_dict,batch_size = optimization_batch_size)

        #print('optimizing...')

        actor_losses = []
        critic_losses = []
        losses = []

        for state_batch, action_batch, advantage_batch, returns_batch, log_prob_batch in zip(samples['state'], samples['action'], samples['advantage'], samples['monte_carlo'], samples['log_prob']):
            with tf.GradientTape() as tape:                
                #print('ACTION:\n',action_batch)
                # Old policy
                old_log_prob = log_prob_batch
                #print('OLD_LOGPROB:\n',old_log_prob)
                # New policy
                new_log_prob, entropy = agent.flowing_log_prob(state_batch,action_batch, return_entropy=True)
                #print('NEW_LOGPROB:\n',new_log_prob)
                ratio = tf.exp(new_log_prob - tf.cast(old_log_prob, tf.float32))
                #print('RATIO:\n',ratio)
                #print('ADV:\n',advantage_batch)
                ppo1 = ratio * tf.expand_dims(advantage_batch,1)
                #print('PPO1:\n',ppo1)
                ppo2 = tf.clip_by_value(ratio, 1-clipping_value, 1+clipping_value) * tf.expand_dims(advantage_batch,1)
                #print('PPO2:\n',ppo2)
                actor_loss = -tf.reduce_mean(tf.minimum(ppo1,ppo2),0)
                #print('ACTOR_LOSS:\n',actor_loss)

                value_target = returns_batch
                #print('VALUE_TARGET:\n',value_target)
                value_pred = agent.v(state_batch)
                #print('VALUE_PRED:\n',value_pred)
                critic_loss = mse_loss(value_target,value_pred)
                #print('CRITIC_LOSS:\n',critic_loss)

                #print('ENTROPY:\n',entropy)
                total_loss = actor_loss + critic_discount * critic_loss - entropy_beta * entropy
                #print('TOTAL_LOSS:\n',total_loss)

                # policy_weights = [var for var in manager.get_agent().model.trainable_variables if 'Policy' in var.name]
                # value_weights = [var for var in manager.get_agent().model.trainable_variables if 'Value' in var.name]
                gradients = tape.gradient(total_loss, agent.model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
            
            actor_losses.append(actor_loss)                
            critic_losses.append(critic_loss)                
            losses.append(total_loss)                

        # Set new weights
        manager.set_agent(agent.get_weights())

        #print('TEST')

        # Update aggregator
        steps, current_rewards = manager.test(
            max_steps=1000,
            test_episodes=10,
            render=False,
            evaluation_measure="time_and_reward",
            )

        #if (e+1) % 5 == 0:
        manager.test(
            max_steps=1000,
            test_episodes=1,
            render=True
            )
        manager.update_aggregator(loss=losses, reward=current_rewards, time=steps)
        
        # Collect all rewards
        rewards.extend(current_rewards)
        # Average reward over last 100 episodes
        avg_reward = sum(rewards[-100:])/min(len(rewards),100)

        # Print progress
        print(
            f"epoch ::: {e}  loss ::: {np.mean(losses):.3f}   avg_current_reward ::: {np.mean(current_rewards):.3f}   avg_reward ::: {avg_reward:.3f}   avg_timesteps ::: {np.mean(steps):.2f}"
        )

        if avg_reward > env.spec.reward_threshold:
            print(f'\n\nEnvironment solved after {e+1} episodes!')
            # Save model
            manager.save_model(saving_path, e, model_name='LunarLanderContinuous')
            break

    print("testing optimized agent")
    manager.test(
        max_steps=1000,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )