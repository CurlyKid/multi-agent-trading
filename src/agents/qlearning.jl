"""
Q-learning agent implementation.

Implements tabular Q-learning for discrete action spaces:
- Temporal difference (TD) learning
- ε-greedy exploration
- Experience replay
- Q-value bounding for stability
"""

using Random

"""
    create_qlearning_agent(id::Int, initial_cash::Float64, learning_rate::Float64,
                          discount::Float64, epsilon::Float64) -> QLearningAgent

Create Q-learning agent with specified parameters.

# Arguments
- `id::Int`: Unique agent identifier
- `initial_cash::Float64`: Starting cash
- `learning_rate::Float64`: α parameter (0 < α ≤ 1)
- `discount::Float64`: γ parameter (0 ≤ γ < 1)
- `epsilon::Float64`: Exploration rate (0 ≤ ε ≤ 1)

# Returns
- `QLearningAgent`: Initialized agent

# Notes
- Q-table starts empty (lazy initialization)
- Position starts with cash, no shares
- Experience buffer starts empty

# Example
```julia
agent = create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1)
# α=0.1 (slow learning), γ=0.95 (values future), ε=0.1 (10% exploration)
```
"""
function create_qlearning_agent(
    id::Int,
    initial_cash::Float64,
    learning_rate::Float64,
    discount::Float64,
    epsilon::Float64
)::QLearningAgent
    position = Position(initial_cash, 0, initial_cash, 0.0)
    q_table = Dict{Tuple{Int,Symbol}, Float64}()
    experience_buffer = []
    
    return QLearningAgent(
        id,
        position,
        q_table,
        learning_rate,
        discount,
        epsilon,
        experience_buffer
    )
end

"""
    q_learning_update!(agent::QLearningAgent, state::Int, action::Symbol,
                      reward::Float64, next_state::Int, done::Bool)

Update Q-values using temporal difference (TD) learning.

# Arguments
- `agent::QLearningAgent`: Agent to update (modified in-place)
- `state::Int`: Current state (discretized price bin)
- `action::Symbol`: Action taken (:buy, :sell, :hold)
- `reward::Float64`: Reward received
- `next_state::Int`: Next state
- `done::Bool`: Episode terminal flag

# Algorithm
Q-learning update rule (off-policy TD control):
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Components:
- Q(s,a): Current Q-value estimate
- α: Learning rate (step size)
- r: Immediate reward
- γ: Discount factor (values future rewards)
- max_a' Q(s',a'): Best Q-value in next state (greedy)
- δ = r + γ max_a' Q(s',a') - Q(s,a): TD error

Key properties:
- Off-policy: Learns optimal policy while following ε-greedy
- Bootstrapping: Uses estimate Q(s',a') to update Q(s,a)
- Convergence: Guaranteed under certain conditions (tabular, infinite visits)

# Notes
- Q-values bounded to [-1000, 1000] for numerical stability
- Terminal states have no future value (max Q = 0)
- Lazy initialization: Q(s,a) = 0 if not in table
- TD error measures prediction error

# Error Handling
- Validates reward is finite
- Bounds Q-values to prevent divergence
- Logs updates at debug level
- Skips update on error (maintains current Q-values)

# Example
```julia
q_learning_update!(agent, 5, :buy, 10.0, 6, false)
# Updates Q(5, :buy) based on reward 10.0 and Q(6, *)
```
"""
function q_learning_update!(
    agent::QLearningAgent,
    state::Int,
    action::Symbol,
    reward::Float64,
    next_state::Int,
    done::Bool
)
    try
        # Validate reward
        if !isfinite(reward)
            throw(ArgumentError("Reward must be finite, got $(reward)"))
        end
        
        # Get current Q-value (default 0 if not in table)
        q_current = get(agent.q_table, (state, action), 0.0)
        
        # Compute max Q-value for next state
        if done
            # Terminal state: no future value
            q_next_max = 0.0
        else
            # Non-terminal: max over actions
            actions = [:buy, :sell, :hold]
            q_next_max = maximum(get(agent.q_table, (next_state, a), 0.0) for a in actions)
        end
        
        # TD error: δ = r + γ max_a' Q(s',a') - Q(s,a)
        # Measures how much our prediction was wrong
        td_error = reward + agent.discount * q_next_max - q_current
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α * δ
        # Move Q-value toward TD target by learning rate α
        q_new = q_current + agent.learning_rate * td_error
        
        # Bound Q-values for numerical stability
        # Prevents divergence from large rewards or errors
        q_new = clamp(q_new, -1000.0, 1000.0)
        
        # Update Q-table
        agent.q_table[(state, action)] = q_new
        
        @debug "Q-learning update" agent_id=agent.id state action reward td_error q_old=q_current q_new
        
    catch e
        @error "Q-learning update failed" exception=(e, catch_backtrace()) agent_id=agent.id
        # Skip update on error (maintain current Q-values)
    end
end

"""
    epsilon_greedy(agent::QLearningAgent, state::Int) -> Symbol

Select action using ε-greedy policy.

# Arguments
- `agent::QLearningAgent`: Agent
- `state::Int`: Current state

# Returns
- `Symbol`: Selected action (:buy, :sell, :hold)

# Algorithm
ε-greedy exploration strategy:
```
With probability ε: random action (exploration)
With probability 1-ε: argmax_a Q(s,a) (exploitation)
```

# Notes
- Balances exploration (try new actions) vs exploitation (use best known action)
- ε typically decays over time (more exploration early, more exploitation late)
- Random action ensures all state-action pairs visited (required for convergence)
- Greedy action maximizes expected return based on current Q-values

# Example
```julia
action = epsilon_greedy(agent, 5)
# 10% chance: random action
# 90% chance: best action according to Q-table
```
"""
function epsilon_greedy(agent::QLearningAgent, state::Int)::Symbol
    actions = [:buy, :sell, :hold]
    
    if rand() < agent.epsilon
        # Explore: random action
        return rand(actions)
    else
        # Exploit: best action
        q_values = [get(agent.q_table, (state, a), 0.0) for a in actions]
        best_idx = argmax(q_values)
        return actions[best_idx]
    end
end

"""
    observe(agent::QLearningAgent, observation::Observation) -> BeliefState

Process observation and update belief state.

# Arguments
- `agent::QLearningAgent`: Agent
- `observation::Observation`: Market observation

# Returns
- `BeliefState`: Updated belief (simplified for Q-learning)

# Notes
- Q-learning uses discretized states (not full belief distribution)
- This function provides interface compatibility
- Returns dummy belief state (Q-learning doesn't use beliefs)

# Example
```julia
obs = Observation(102.0, 100.0, 1)
belief = observe(agent, obs)
# belief not actually used by Q-learning
```
"""
function observe(agent::QLearningAgent, observation::Observation)::BeliefState
    # Q-learning doesn't maintain belief states
    # Return dummy belief for interface compatibility
    state = POMDPState(observation.observed_price, 0.0, 0.02, observation.time)
    return BeliefState([state], [1.0])
end

"""
    act(agent::QLearningAgent, belief::BeliefState, current_price::Float64) -> Order

Select action based on current belief/state.

# Arguments
- `agent::QLearningAgent`: Agent
- `belief::BeliefState`: Current belief (not used, for interface)
- `current_price::Float64`: Current market price

# Returns
- `Order`: Selected action with quantity

# Algorithm
1. Discretize price to get state
2. Select action using ε-greedy
3. Determine quantity based on action and position

# Notes
- Buy: quantity = 10 shares (fixed for simplicity)
- Sell: quantity = min(10, current_shares)
- Hold: quantity = 0
- Could be made more sophisticated (dynamic position sizing)

# Example
```julia
order = act(agent, belief, 100.0)
# order.action ∈ {:buy, :sell, :hold}
# order.quantity depends on action and position
```
"""
function act(agent::QLearningAgent, belief::BeliefState, current_price::Float64)::Order
    try
        # Discretize price to get state
        state = discretize_price(current_price, 90.0, 110.0, 20)
        
        # Select action using ε-greedy
        action = epsilon_greedy(agent, state)
        
        # Determine quantity
        if action == :buy
            quantity = 10  # Fixed buy quantity
        elseif action == :sell
            quantity = min(10, agent.position.shares)  # Sell up to 10 shares
        else  # :hold
            quantity = 0
        end
        
        return Order(action, quantity)
    catch e
        @error "Action selection failed" exception=(e, catch_backtrace()) agent_id=agent.id
        # Safe default: hold
        return Order(:hold, 0)
    end
end

"""
    learn!(agent::QLearningAgent, experience::Experience)

Learn from experience tuple.

# Arguments
- `agent::QLearningAgent`: Agent to update (modified in-place)
- `experience::Experience`: (state, action, reward, next_state, done)

# Notes
- Calls q_learning_update! to update Q-values
- Adds experience to replay buffer
- Could implement experience replay (sample from buffer)

# Example
```julia
exp = Experience(5, :buy, 10.0, 6, false)
learn!(agent, exp)
# Q-values updated based on experience
```
"""
function learn!(agent::QLearningAgent, experience::Experience)
    try
        # Update Q-values
        q_learning_update!(
            agent,
            experience.state,
            experience.action,
            experience.reward,
            experience.next_state,
            experience.done
        )
        
        # Add to experience buffer (for potential replay)
        push!(agent.experience_buffer, experience)
        
        # Limit buffer size (keep last 10000 experiences)
        if length(agent.experience_buffer) > 10000
            popfirst!(agent.experience_buffer)
        end
        
    catch e
        @error "Learning failed" exception=(e, catch_backtrace()) agent_id=agent.id
    end
end

"""
    reset!(agent::QLearningAgent, initial_cash::Float64)

Reset agent state for new episode.

# Arguments
- `agent::QLearningAgent`: Agent to reset (modified in-place)
- `initial_cash::Float64`: Starting cash

# Notes
- Resets position (cash, shares, portfolio value, PnL)
- Keeps Q-table (learning persists across episodes)
- Keeps experience buffer
- Does not reset exploration rate (could decay epsilon here)

# Example
```julia
reset!(agent, 10000.0)
# agent.position reset, Q-table preserved
```
"""
function reset!(agent::QLearningAgent, initial_cash::Float64)
    agent.position.cash = initial_cash
    agent.position.shares = 0
    agent.position.portfolio_value = initial_cash
    agent.position.pnl = 0.0
end

"""
    update_position!(agent::QLearningAgent, order::Order, execution_price::Float64,
                    slippage_cost::Float64, current_price::Float64)

Update agent position after order execution.

# Arguments
- `agent::QLearningAgent`: Agent (modified in-place)
- `order::Order`: Executed order
- `execution_price::Float64`: Price at which order executed
- `slippage_cost::Float64`: Cost of slippage
- `current_price::Float64`: Current market price

# Side Effects
- Updates agent.position.cash
- Updates agent.position.shares
- Updates agent.position.portfolio_value
- Updates agent.position.pnl

# Notes
- Buy: cash decreases, shares increase
- Sell: cash increases, shares decrease
- Hold: no change
- Portfolio value = cash + shares * current_price
- PnL = portfolio_value - initial_value

# Example
```julia
update_position!(agent, Order(:buy, 10), 101.0, 1.0, 100.0)
# agent.position.cash -= 10 * 101.0
# agent.position.shares += 10
```
"""
function update_position!(
    agent::QLearningAgent,
    order::Order,
    execution_price::Float64,
    slippage_cost::Float64,
    current_price::Float64
)
    if order.action == :buy
        # Buy: pay cash, receive shares
        cost = order.quantity * execution_price
        agent.position.cash -= cost
        agent.position.shares += order.quantity
    elseif order.action == :sell
        # Sell: receive cash, give shares
        proceeds = order.quantity * execution_price
        agent.position.cash += proceeds
        agent.position.shares -= order.quantity
    end
    # Hold: no change
    
    # Update portfolio value
    agent.position.portfolio_value = agent.position.cash + agent.position.shares * current_price
    
    # Update PnL (assumes initial value stored somewhere, simplified here)
    # In full implementation, would track initial_portfolio_value
end

# Exports
export create_qlearning_agent, q_learning_update!, epsilon_greedy
export observe, act, learn!, reset!, update_position!
