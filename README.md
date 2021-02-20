# ReAllY

Your Reinforcement Learning Ally.

A framework based on tensorflow, RAY and gym to handle reinforcement learning tasks.



## General Remarks

This framework was originally build for the block course "Deep Reinforcement Learning" at the University of Osnabrück.
In the block course students are asked to implement standard (deep) RL algorithms with the help of the framework. Sample solutions will be published onec the course is done.

This framework is still under construction and can yet be optimized. If you run into errors or see ways to make something more efficient, please feel free to raise an issue or contact me dircetly (Charlie Lange, chalange@uos.de) and help make this framework better for everyon!

## General Design

-> insert graphic

### Sample Mnager
The sample manager manages collecting experience using remote runners. Therfore it has to be initialized with the specifications for the environment and the model used, how the agent should behave and what data needs to be collected. 

#### Buffer
Using the sample manager, a buffer can be initialized in which the data the sample manager collects via its remote runners can be stored and from which the user can sample data.

#### Aggregator 
Using the sample manager, an evaluation aggregator can be initialized where the training progress from the main process can be stored. This aggregator already computes smooth averages and handles plotting.


## Usage
### Sample Manager
The sample manager should be initalized from the main process.

#### Args
    **model**: callable model Object which needs to have the methods 'load_weights()' and 'set_weights()'
    **environment**: string specifying gym environment or object of custom gym-like (implementing the same methods) environment
    **num_parallel**: int, number of how many agents to run in parall (should correspond to machines cpu cores)
    **total_steps**: int, size of the total steps collected
    **returns**: list of strings specifying what is to be returned by the box, supported are: 'value_estimate', 'log_prob', 'monte_carlo'
    **actin_sampling_type**: string, type of sampling actions, supported are 'epsilon_greedy', 'thompson', or 'continous_normal_diagonal'

#### kwargs:
        **model_kwargs**: dict, optional model initialization specifications
        **weights**: optional, weights which can be loaded into the agent for remote data collecting
        **input_shape**: shape or boolean (if shape not needed for first call of model), defaults shape of the environments reset state
        
         **env_config**: dict, opitonal configurations for environment creation if a custom environment is used
        
        **num_episodes**: specifies the total number of episodes to run on the environment for each runner, defaults to 1
        **num_steps**: specifies the total number of steps to run on the environment for each runner
      
        **gamma**: float, discount factor for monte carlo return, defaults to 0.99
        **temperature**: float, temperature for thomson sampling, defaults to 1
        **epsilon**: epsilon for epsilon greedy sampling, defaults to 0.95
       
        **remote_min_returns**: int, minimum number of remote runner results to wait for, defaults to 10% of num_parallel
        **remote_time_out**: float, maximum amount of time (in seconds) to wait on the remote runner results, defaults to None
       
    """


#### Methods

