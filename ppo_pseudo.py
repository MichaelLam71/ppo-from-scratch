
"""
make actor network (state → Gaussian distribution over actions)
make critic network (state → value estimate)

loop many iterations:

    # COLLECT EXPERIENCE
    empty list of trajectories
    for 2048 steps:
        get state from environment
        pass state through actor → get Gaussian distribution
        sample action from distribution
        record log_prob of that action
        pass state through critic → get value estimate
        take action in environment → get reward, next_state, done
        store (state, action, log_prob, reward, value, done)
    
    # COMPUTE ADVANTAGES
    for each timestep in trajectories, working backwards:
        if done:
            future_value = 0
        else:
            future_value = critic(next_state)
        advantage = reward + gamma * future_value - value
    
    normalize advantages to mean 0, std 1
    compute returns = advantages + values
    
    # UPDATE NETWORKS (reuse the same batch multiple times)
    for 3-4 epochs:
        for each mini-batch from the trajectories:
            
            # actor update
            pass states through actor → get new distribution
            compute new_log_prob of the stored actions
            ratio = exp(new_log_prob - old_log_prob)
            clipped_ratio = clip(ratio, 1-0.2, 1+0.2)
            actor_loss = -min(ratio * advantage, clipped_ratio * advantage)
            
            # critic update
            new_value = critic(state)
            critic_loss = (new_value - return)^2
            
            # update both networks
            backpropagate actor_loss + critic_loss
            gradient descent step
    
    throw away all trajectories

    """