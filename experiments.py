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

from ppo_model import A2C, TargetNetwork

tf.random.set_seed(42)

if __name__ == "__main__":

    bipedal = "BipedalWalker-v3"
    lunar = "LunarLanderContinuous-v2"

    for e_name in [bipedal, lunar]:
        if not os.path.exists("./"+e_name):
            os.makedirs(e_name)

    ### this is the main setting we change between experiments ###
    env_name = bipedal  # choose env name, either bipedal or lunar
    use_rnd = False     # whether to use vanilla PPO or RND
    
    env = gym.make(env_name)
    env.seed(42)

    if env_name == bipedal:
        model_kwargs = {"layers": [48,48,48], "action_dim": env.action_space.shape[0]}
    else:
        model_kwargs = {"layers": [32,32,32], "action_dim": env.action_space.shape[0]}
    
    learning_rate = 0.001
    max_episodes = 300
    sampled_batches = 512
    optimization_batch_size = 64
    gamma = 0.99
    my_lambda = 0.95
    clipping_value = 0.3   
    critic_discount = 0.5
    entropy_beta = 0.001
    if use_rnd:
        curiosity_weight = 0.1

    kwargs = {
        "model": A2C,
        "environment": env_name, 
        "num_parallel": 3,
        "total_steps": 420,
        "returns": ['value_estimate', 'log_prob', 'monte_carlo'],
        "model_kwargs": model_kwargs,
        "action_sampling_type": "continuous_normal_diagonal",
        #"gamma": gamma
    }

    ### RND ###
    if use_rnd:
        layers = [8,8]
        k = 8
        target_network = TargetNetwork(layers, k) # fixed random net used to obtain features
        predictor = TargetNetwork(layers, k) # predictor network we will train to match target network

        pred_optimizer = tf.keras.optimizers.Adam(learning_rate)


    # Initialize the loss function
    mse_loss = tf.keras.losses.MeanSquaredError()

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)


    # Initialize
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    saving_path = os.getcwd() + "/" + env_name

    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", 'reward', 'time']
    )

    rewards = []

    agent = manager.get_agent()

    with open(env_name+'/results.csv','a') as fd:
        fd.write('epoch,loss,reward,steps\n')

    print('TRAINING')
    for e in range(max_episodes):
        
        # Sample data to optimize
        sample_dict = manager.sample(
            sample_size = sampled_batches*optimization_batch_size,
            from_buffer = False
            )
        
        # Compute Advantages
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

        ### RND ###
        if use_rnd:
            # calculate features from environment observations
            sample_dict['features'] = []
            for state in sample_dict['state']:
                state = np.array(state)
                state = np.reshape(state, (1,env.observation_space.shape[0]))
                sample_dict['features'].append(tf.squeeze(target_network(state)))

        # Remove keys that are no longer used
        sample_dict.pop('value_estimate')
        sample_dict.pop('state_new')
        sample_dict.pop('reward')
        sample_dict.pop('not_done')

        samples = dict_to_dict_of_datasets(sample_dict,batch_size = optimization_batch_size)

        #print('optimizing...')

        actor_losses = []
        critic_losses = []
        rnd_losses = []
        losses = []
        advantages = []

        for state_batch, action_batch, advantage_batch, returns_batch, log_prob_batch, feature_batch in zip(samples['state'], samples['action'], samples['advantage'], samples['monte_carlo'], samples['log_prob'], samples['features']):
            
            if use_rnd:
                with tf.GradientTape() as tape:
                    feature_pred = predictor(state_batch)
                    rnd_loss = tf.keras.losses.MSE(feature_batch, feature_pred)

                    rnd_gradients = tape.gradient(rnd_loss, predictor.trainable_variables)
                pred_optimizer.apply_gradients(zip(rnd_gradients, predictor.trainable_variables))

            with tf.GradientTape() as tape:               

                ### RND ### 
                if use_rnd:
                    advantage_batch += curiosity_weight * rnd_loss # incorporate rnd loss into reward signal
                
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

                gradients = tape.gradient(total_loss, agent.model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

            actor_losses.append(actor_loss)                
            critic_losses.append(critic_loss)   
            ### RND ###
            if use_rnd:
                rnd_losses.append(rnd_loss)             
            losses.append(total_loss)       
            advantages.append(tf.reduce_mean(advantage_batch))         

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
            f"e: {e} loss: {np.mean(losses):.3f} RND loss: {np.mean(rnd_losses):.6f}  adv: {np.mean(advantages):.6f} avg_curr_rew: {np.mean(current_rewards):.3f}  avg_reward: {avg_reward:.3f}  avg_steps: {np.mean(steps):.2f}"
        )

        # print progress to file
        with open(env_name+'/results.csv','a') as fd:
            fd.write(','.join([str(np.mean(x)) for x in [e, losses, current_rewards, steps]])+'\n')

        if (e % 25) == 0:#avg_reward > env.spec.reward_threshold:
            #print(f'\n\nEnvironment solved after {e+1} episodes!')
            print("Saving Model")
            # Save model
            manager.save_model(saving_path, e, model_name='model')
            break

    print("testing optimized agent")
    manager.test(
        max_steps=1000,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )