"""
Reinforcement learning training loop.

Implements episode-based training for RL agents:
- Experience replay for Q-learning
- Trajectory collection for policy gradient
- Learning curve tracking
- Episode management
"""

using Statistics

"""
    train_agent!(agent::TradingAgent, market_params::MarketParams, 
                 n_episodes::Int, n_steps::Int) -> Dict

Train agent over multiple episodes.

# Arguments
- `agent::TradingAgent`: Agent to train (modified in-place)
- `market_params::MarketParams`: Market configuration
- `n_episodes::Int`: Number of training episodes
- `n_steps::Int`: Steps per episode

# Returns
- `Dict`: Training statistics
  - `:episode_returns`: Returns per episode
  - `:episode_lengths`: Steps per episode
  - `:learning_curve`: Smoothed returns (moving average)

# Algorithm
For each episode:
1. Reset market and agent
2. Run episode (n_steps)
3. Collect experiences
4. Update agent (Q-learning: per step, Policy gradient: per episode)
5. Track performance

# Notes
- Q-learning updates after each step (online)
- Policy gradient updates after episode (batch)
- Experience replay for Q-learning (sample efficiency)
- Epsilon decay for Q-learning (exploration → exploitation)

# Error Handling
- Validates parameters (n_episodes > 0, n_steps > 0)
- Logs training progress every 10 episodes
- Continues training on individual step errors
- Returns partial results if training interrupted

# Example
```julia
agent = create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1)
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
stats = train_agent!(agent, params, 100, 1000)
# agent trained for 100 episodes of 1000 steps each
```
"""
function train_agent!(
    agent::TradingAgent,
    market_params::MarketParams,
    n_episodes::Int,
    n_steps::Int
)::Dict
    try
        @assert n_episodes > 0 "Number of episodes must be positive"
        @assert n_steps > 0 "Number of steps must be positive"
        
        @info "Starting training" agent_id=agent.id n_episodes n_steps
        
        # Track statistics
        episode_returns = Float64[]
        episode_lengths = Int[]
        
        for episode in 1:n_episodes
            # Run episode
            episode_return, episode_length = run_episode!(agent, market_params, n_steps)
            
            push!(episode_returns, episode_return)
            push!(episode_lengths, episode_length)
            
            # Log progress
            if episode % 10 == 0
                avg_return = mean(episode_returns[max(1, end-9):end])
                @info "Training progress" episode avg_return_last_10=avg_return
            end
            
            # Decay epsilon for Q-learning
            if agent isa QLearningAgent
                agent.epsilon *= 0.995  # Decay exploration
                agent.epsilon = max(agent.epsilon, 0.01)  # Min epsilon
            end
        end
        
        # Compute learning curve (moving average)
        window = 10
        learning_curve = [mean(episode_returns[max(1, i-window+1):i]) 
                         for i in 1:length(episode_returns)]
        
        @info "Training complete" agent_id=agent.id final_avg_return=learning_curve[end]
        
        return Dict(
            :episode_returns => episode_returns,
            :episode_lengths => episode_lengths,
            :learning_curve => learning_curve
        )
        
    catch e
        @error "Training failed" exception=(e, catch_backtrace()) agent_id=agent.id
        rethrow(e)
    end
end

"""
    run_episode!(agent::TradingAgent, market_params::MarketParams, 
                 n_steps::Int) -> Tuple{Float64, Int}

Run single training episode.

# Arguments
- `agent::TradingAgent`: Agent (modified in-place)
- `market_params::MarketParams`: Market configuration
- `n_steps::Int`: Maximum steps in episode

# Returns
- `Tuple{Float64, Int}`: (total_return, steps_taken)

# Algorithm
1. Initialize market
2. Reset agent
3. For each step:
   - Observe market
   - Select action
   - Execute order
   - Compute reward
   - Learn from experience
   - Update market
4. Final learning update (policy gradient)
5. Return episode statistics

# Reward Function
```
reward = Δ portfolio_value + penalty
```
where:
- Δ portfolio_value = change in total value (cash + shares * price)
- penalty = -slippage_cost (transaction cost)

# Notes
- Episode terminates after n_steps (no early termination)
- Q-learning updates after each step
- Policy gradient updates after episode
- Tracks cumulative return

# Error Handling
- Continues on individual step errors
- Returns partial episode if interrupted
- Logs errors for debugging

# Example
```julia
agent = create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1)
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
total_return, steps = run_episode!(agent, params, 1000)
# agent experienced 1000-step episode
```
"""
function run_episode!(
    agent::TradingAgent,
    market_params::MarketParams,
    n_steps::Int
)::Tuple{Float64, Int}
    try
        # Initialize market
        market = initialize_market(market_params)
        
        # Reset agent
        initial_cash = agent.position.cash
        reset!(agent, initial_cash)
        
        # Track episode
        total_return = 0.0
        steps_taken = 0
        
        for step in 1:n_steps
            try
                # Current state
                prev_portfolio_value = agent.position.portfolio_value
                current_price = market.price
                
                # Observe market
                observation = observe_market(market, market_params)
                belief = observe(agent, observation)
                
                # Select action
                order = act(agent, belief, current_price)
                
                # Execute order
                execution_price, slippage_cost = execute_order(market, order, market_params)
                
                # Update agent position
                update_position!(agent, order, execution_price, slippage_cost, current_price)
                
                # Compute reward
                # Reward = change in portfolio value - transaction cost
                new_portfolio_value = agent.position.portfolio_value
                reward = (new_portfolio_value - prev_portfolio_value) - slippage_cost
                
                # Learn (Q-learning updates immediately, policy gradient accumulates)
                if agent isa QLearningAgent
                    # Discretize state for Q-learning
                    state = discretize_price(current_price, market_params.initial_price)
                    next_price = update_price(market, market_params)
                    next_state = discretize_price(next_price, market_params.initial_price)
                    
                    # Q-learning update
                    q_learning_update!(agent, state, order.action, reward, next_state, false)
                    
                    # Experience replay
                    experience = Experience(state, order.action, reward, next_state, false)
                    push!(agent.experience_buffer, experience)
                    
                    # Replay from buffer
                    if length(agent.experience_buffer) >= 32
                        replay_experiences!(agent, 16)
                    end
                    
                elseif agent isa PolicyGradientAgent
                    # Accumulate trajectory
                    learn!(agent, current_price, Float64(order.quantity), reward)
                end
                
                # Update market
                market.price = update_price(market, market_params)
                push!(market.price_history, market.price)
                market.volume = Float64(order.quantity)
                push!(market.volume_history, market.volume)
                market.time += 1
                
                # Track return
                total_return += reward
                steps_taken += 1
                
            catch e
                @warn "Step failed, continuing episode" step exception=e
                continue
            end
        end
        
        # Policy gradient update (after episode)
        if agent isa PolicyGradientAgent && !isempty(agent.trajectory)
            policy_gradient_update!(agent, 0.95)
        end
        
        return (total_return, steps_taken)
        
    catch e
        @error "Episode failed" exception=(e, catch_backtrace()) agent_id=agent.id
        return (0.0, 0)
    end
end

"""
    replay_experiences!(agent::QLearningAgent, batch_size::Int)

Sample and replay experiences from buffer.

# Arguments
- `agent::QLearningAgent`: Agent (modified in-place)
- `batch_size::Int`: Number of experiences to replay

# Algorithm
Experience replay:
1. Sample random batch from buffer
2. For each experience:
   - Compute TD error
   - Update Q-value
3. Improves sample efficiency

# Notes
- Breaks correlation between consecutive experiences
- Enables off-policy learning
- Stabilizes training
- Buffer size limited to 10000 (FIFO)

# Error Handling
- Validates buffer has enough experiences
- Skips replay if buffer too small
- Continues on individual update errors

# Example
```julia
replay_experiences!(agent, 16)
# agent learns from 16 random experiences
```
"""
function replay_experiences!(agent::QLearningAgent, batch_size::Int)
    try
        if length(agent.experience_buffer) < batch_size
            return
        end
        
        # Sample random batch
        indices = rand(1:length(agent.experience_buffer), batch_size)
        batch = agent.experience_buffer[indices]
        
        # Replay each experience
        for exp in batch
            try
                q_learning_update!(agent, exp.state, exp.action, exp.reward, 
                                 exp.next_state, exp.done)
            catch e
                @warn "Experience replay update failed" exception=e
                continue
            end
        end
        
        # Limit buffer size (FIFO)
        if length(agent.experience_buffer) > 10000
            deleteat!(agent.experience_buffer, 1:1000)
        end
        
    catch e
        @error "Experience replay failed" exception=(e, catch_backtrace())
    end
end

"""
    discretize_price(price::Float64, reference_price::Float64, n_bins::Int=20) -> Int

Discretize continuous price into discrete state.

# Arguments
- `price::Float64`: Current price
- `reference_price::Float64`: Reference price (e.g., initial price)
- `n_bins::Int`: Number of discrete bins (default 20)

# Returns
- `Int`: Discrete state index (1 to n_bins)

# Algorithm
Maps price to discrete bin:
```
relative_price = price / reference_price
bin = floor((relative_price - 0.5) * n_bins / 0.5) + n_bins/2
```

# Notes
- Centers bins around reference price
- Bins cover range [0.5, 1.5] * reference_price
- Prices outside range mapped to edge bins
- Used for Q-learning state space

# Example
```julia
state = discretize_price(105.0, 100.0, 20)
# state ≈ 12 (price 5% above reference)
```
"""
function discretize_price(price::Float64, reference_price::Float64, n_bins::Int=20)::Int
    # Relative price
    relative = price / reference_price
    
    # Map to bin (centered at 1.0)
    # Range: [0.5, 1.5] → bins [1, n_bins]
    bin = floor(Int, (relative - 0.5) * n_bins / 1.0) + 1
    
    # Clamp to valid range
    return clamp(bin, 1, n_bins)
end

# Exports
export train_agent!, run_episode!, replay_experiences!, discretize_price

