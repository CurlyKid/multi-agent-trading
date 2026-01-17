"""
Policy gradient agent implementation.

Implements REINFORCE algorithm for continuous action spaces:
- Policy gradient estimation
- Baseline for variance reduction
- Gradient clipping for stability
- Trajectory-based learning
"""

using Random
using LinearAlgebra

"""
    create_policy_gradient_agent(id::Int, initial_cash::Float64, n_params::Int,
                                 learning_rate::Float64) -> PolicyGradientAgent

Create policy gradient agent with specified parameters.

# Arguments
- `id::Int`: Unique agent identifier
- `initial_cash::Float64`: Starting cash
- `n_params::Int`: Number of policy parameters
- `learning_rate::Float64`: Step size for gradient updates

# Returns
- `PolicyGradientAgent`: Initialized agent

# Notes
- Policy parameters initialized randomly (small values)
- Baseline starts at 0 (updated during training)
- Trajectory starts empty (filled during episode)

# Example
```julia
agent = create_policy_gradient_agent(1, 10000.0, 10, 0.01)
# 10 policy parameters, learning rate 0.01
```
"""
function create_policy_gradient_agent(
    id::Int,
    initial_cash::Float64,
    n_params::Int,
    learning_rate::Float64
)::PolicyGradientAgent
    position = Position(initial_cash, 0, initial_cash, 0.0)
    policy_params = randn(n_params) * 0.1  # Small random initialization
    baseline = 0.0
    trajectory = []
    
    return PolicyGradientAgent(
        id,
        position,
        policy_params,
        baseline,
        learning_rate,
        trajectory
    )
end

"""
    compute_returns(trajectory::Vector{Tuple}, discount::Float64) -> Vector{Float64}

Compute discounted returns from trajectory.

# Arguments
- `trajectory::Vector{Tuple}`: Episode trajectory (state, action, reward)
- `discount::Float64`: Discount factor γ

# Returns
- `Vector{Float64}`: Discounted returns G_t for each timestep

# Algorithm
Discounted return (computed backwards for efficiency):
```
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ... + γ^(T-t) r_T
```

Recursive formulation:
```
G_T = r_T
G_t = r_t + γ G_{t+1}
```

# Notes
- Computed backwards (from end to start) for efficiency
- Measures total future reward from each timestep
- Discount γ < 1 makes near-term rewards more valuable
- Used in REINFORCE to weight policy gradient

# Example
```julia
trajectory = [(1, 0.5, 1.0), (2, 0.3, 2.0), (3, 0.7, 3.0)]
returns = compute_returns(trajectory, 0.95)
# returns[1] = 1.0 + 0.95*2.0 + 0.95²*3.0 ≈ 5.61
# returns[2] = 2.0 + 0.95*3.0 ≈ 4.85
# returns[3] = 3.0
```
"""
function compute_returns(trajectory::Vector{Tuple}, discount::Float64)::Vector{Float64}
    n = length(trajectory)
    returns = zeros(n)
    G = 0.0
    
    # Compute backwards: G_t = r_t + γ G_{t+1}
    for t in n:-1:1
        _, _, reward = trajectory[t]
        G = reward + discount * G
        returns[t] = G
    end
    
    return returns
end

"""
    policy_gradient_update!(agent::PolicyGradientAgent, discount::Float64=0.95)

Update policy parameters using REINFORCE algorithm.

# Arguments
- `agent::PolicyGradientAgent`: Agent to update (modified in-place)
- `discount::Float64`: Discount factor γ (default 0.95)

# Algorithm
REINFORCE policy gradient:
```
∇J(θ) = E[∇log π(a|s,θ) * (G - b)]
```

Components:
- θ: Policy parameters
- π(a|s,θ): Policy (probability of action a in state s)
- G: Discounted return (total future reward)
- b: Baseline (reduces variance without bias)
- ∇log π: Score function (direction to increase probability)

Update rule:
```
θ ← θ + α * ∇J(θ)
```

Key insights:
- Increases probability of actions with high returns (G > b)
- Decreases probability of actions with low returns (G < b)
- Baseline reduces variance (makes learning more stable)
- Unbiased: E[∇log π * b] = 0 (baseline doesn't change expected gradient)

# Notes
- Baseline updated as moving average of returns
- Gradients clipped to [-10, 10] for stability
- Skips update if gradients non-finite
- Clears trajectory after update

# Error Handling
- Validates trajectory is non-empty
- Ensures gradients are finite
- Clips gradients for stability
- Logs updates at debug level
- Skips update on error

# Example
```julia
# After episode completes
policy_gradient_update!(agent, 0.95)
# Policy parameters updated based on episode returns
# Trajectory cleared for next episode
```
"""
function policy_gradient_update!(agent::PolicyGradientAgent, discount::Float64=0.95)
    try
        if isempty(agent.trajectory)
            @warn "Empty trajectory, skipping policy gradient update" agent_id=agent.id
            return
        end
        
        # Step 1: Compute returns G_t for each timestep
        returns = compute_returns(agent.trajectory, discount)
        
        # Step 2: Update baseline (moving average of returns)
        # Baseline b ≈ E[G] reduces variance without bias
        mean_return = mean(returns)
        agent.baseline = 0.9 * agent.baseline + 0.1 * mean_return
        
        # Step 3: Compute policy gradient
        # ∇J(θ) = Σ_t ∇log π(a_t|s_t,θ) * (G_t - b)
        for (i, (state, action, reward)) in enumerate(agent.trajectory)
            # Advantage: A_t = G_t - b
            # Measures how much better this action was than average
            advantage = returns[i] - agent.baseline
            
            # Compute gradient of log policy
            # For simplicity, using linear policy: π(a|s) ∝ exp(θ^T φ(s,a))
            # ∇log π = φ(s,a) - E[φ(s,a')]
            # Simplified here as feature vector
            grad = compute_policy_gradient(agent, state, action)
            
            # Check gradient is finite
            if !all(isfinite, grad)
                @warn "Non-finite gradient detected, skipping update" agent_id=agent.id
                continue
            end
            
            # Clip gradient for stability
            # Prevents large updates that could destabilize policy
            grad = clamp.(grad, -10.0, 10.0)
            
            # Update parameters: θ ← θ + α * ∇log π * A
            # Increases probability of good actions (A > 0)
            # Decreases probability of bad actions (A < 0)
            agent.policy_params .+= agent.learning_rate .* grad .* advantage
        end
        
        @debug "Policy gradient update" agent_id=agent.id n_steps=length(agent.trajectory) mean_return baseline=agent.baseline
        
        # Clear trajectory for next episode
        empty!(agent.trajectory)
        
    catch e
        @error "Policy gradient update failed" exception=(e, catch_backtrace()) agent_id=agent.id
        # Clear trajectory even on error
        empty!(agent.trajectory)
    end
end

"""
    compute_policy_gradient(agent::PolicyGradientAgent, state::Float64, 
                           action::Float64) -> Vector{Float64}

Compute gradient of log policy.

# Arguments
- `agent::PolicyGradientAgent`: Agent
- `state::Float64`: State (price)
- `action::Float64`: Action taken

# Returns
- `Vector{Float64}`: Gradient ∇log π(a|s,θ)

# Notes
- Simplified implementation (linear features)
- In full implementation, would use neural network
- Returns feature vector weighted by action

# Example
```julia
grad = compute_policy_gradient(agent, 100.0, 0.5)
# grad = feature vector for this state-action pair
```
"""
function compute_policy_gradient(
    agent::PolicyGradientAgent,
    state::Float64,
    action::Float64
)::Vector{Float64}
    # Simplified: use state and action as features
    # In full implementation, would compute proper policy gradient
    n = length(agent.policy_params)
    features = zeros(n)
    features[1] = state / 100.0  # Normalized state
    features[2] = action
    if n > 2
        features[3:end] .= randn(n-2) * 0.01  # Additional features
    end
    return features
end

"""
    observe(agent::PolicyGradientAgent, observation::Observation) -> BeliefState

Process observation and update belief state.

# Arguments
- `agent::PolicyGradientAgent`: Agent
- `observation::Observation`: Market observation

# Returns
- `BeliefState`: Updated belief (simplified)

# Notes
- Policy gradient uses observations directly (not full belief)
- Returns dummy belief for interface compatibility

# Example
```julia
obs = Observation(102.0, 100.0, 1)
belief = observe(agent, obs)
```
"""
function observe(agent::PolicyGradientAgent, observation::Observation)::BeliefState
    # Policy gradient doesn't maintain belief states
    # Return dummy belief for interface compatibility
    state = POMDPState(observation.observed_price, 0.0, 0.02, observation.time)
    return BeliefState([state], [1.0])
end

"""
    act(agent::PolicyGradientAgent, belief::BeliefState, current_price::Float64) -> Order

Select action based on policy.

# Arguments
- `agent::PolicyGradientAgent`: Agent
- `belief::BeliefState`: Current belief (not used)
- `current_price::Float64`: Current market price

# Returns
- `Order`: Selected action with quantity

# Algorithm
1. Compute policy output (action probability/value)
2. Sample action from policy
3. Determine order type and quantity

# Notes
- Stochastic policy (samples from distribution)
- Action space: continuous (order size)
- Simplified: maps to discrete actions for compatibility

# Example
```julia
order = act(agent, belief, 100.0)
# order sampled from policy distribution
```
"""
function act(agent::PolicyGradientAgent, belief::BeliefState, current_price::Float64)::Order
    try
        # Compute policy output
        # Simplified: linear combination of parameters and state
        state_feature = current_price / 100.0
        policy_output = dot(agent.policy_params[1:min(3, length(agent.policy_params))], 
                           [state_feature, 1.0, randn()])
        
        # Map to action
        if policy_output > 0.5
            action = :buy
            quantity = 10
        elseif policy_output < -0.5
            action = :sell
            quantity = min(10, agent.position.shares)
        else
            action = :hold
            quantity = 0
        end
        
        return Order(action, quantity)
    catch e
        @error "Action selection failed" exception=(e, catch_backtrace()) agent_id=agent.id
        return Order(:hold, 0)
    end
end

"""
    learn!(agent::PolicyGradientAgent, state::Float64, action::Float64, reward::Float64)

Add experience to trajectory (learning happens at episode end).

# Arguments
- `agent::PolicyGradientAgent`: Agent (modified in-place)
- `state::Float64`: State
- `action::Float64`: Action taken
- `reward::Float64`: Reward received

# Notes
- Policy gradient learns from full trajectories
- Accumulates (state, action, reward) tuples
- Update happens at episode end (policy_gradient_update!)

# Example
```julia
learn!(agent, 100.0, 0.5, 10.0)
# Experience added to trajectory
# No immediate update (waits for episode end)
```
"""
function learn!(agent::PolicyGradientAgent, state::Float64, action::Float64, reward::Float64)
    try
        push!(agent.trajectory, (state, action, reward))
    catch e
        @error "Learning failed" exception=(e, catch_backtrace()) agent_id=agent.id
    end
end

"""
    reset!(agent::PolicyGradientAgent, initial_cash::Float64)

Reset agent state for new episode.

# Arguments
- `agent::PolicyGradientAgent`: Agent to reset (modified in-place)
- `initial_cash::Float64`: Starting cash

# Notes
- Resets position
- Clears trajectory
- Keeps policy parameters (learning persists)
- Keeps baseline

# Example
```julia
reset!(agent, 10000.0)
# agent.position reset, policy preserved
```
"""
function reset!(agent::PolicyGradientAgent, initial_cash::Float64)
    agent.position.cash = initial_cash
    agent.position.shares = 0
    agent.position.portfolio_value = initial_cash
    agent.position.pnl = 0.0
    empty!(agent.trajectory)
end

"""
    update_position!(agent::PolicyGradientAgent, order::Order, execution_price::Float64,
                    slippage_cost::Float64, current_price::Float64)

Update agent position after order execution.

# Arguments
- `agent::PolicyGradientAgent`: Agent (modified in-place)
- `order::Order`: Executed order
- `execution_price::Float64`: Execution price
- `slippage_cost::Float64`: Slippage cost
- `current_price::Float64`: Current market price

# Side Effects
- Updates position (cash, shares, portfolio value)

# Example
```julia
update_position!(agent, Order(:buy, 10), 101.0, 1.0, 100.0)
```
"""
function update_position!(
    agent::PolicyGradientAgent,
    order::Order,
    execution_price::Float64,
    slippage_cost::Float64,
    current_price::Float64
)
    if order.action == :buy
        cost = order.quantity * execution_price
        agent.position.cash -= cost
        agent.position.shares += order.quantity
    elseif order.action == :sell
        proceeds = order.quantity * execution_price
        agent.position.cash += proceeds
        agent.position.shares -= order.quantity
    end
    
    agent.position.portfolio_value = agent.position.cash + agent.position.shares * current_price
end

# Exports
export create_policy_gradient_agent, compute_returns, policy_gradient_update!
export compute_policy_gradient, observe, act, learn!, reset!, update_position!
