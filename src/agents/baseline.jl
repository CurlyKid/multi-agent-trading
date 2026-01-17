"""
Baseline agent implementations.

Non-learning strategies for comparison:
- Momentum: Buy when price rising, sell when falling
- Mean reversion: Buy when price low, sell when high
- Random: Random actions (control baseline)
"""

using Random

"""
    create_baseline_agent(id::Int, initial_cash::Float64, 
                         strategy::Symbol) -> BaselineAgent

Create baseline agent with specified strategy.

# Arguments
- `id::Int`: Unique agent identifier
- `initial_cash::Float64`: Starting cash
- `strategy::Symbol`: Strategy type (:momentum, :mean_reversion, :random)

# Returns
- `BaselineAgent`: Initialized agent

# Strategies
- `:momentum`: Follow price trends (buy rising, sell falling)
- `:mean_reversion`: Counter price trends (buy low, sell high)
- `:random`: Random actions (control baseline)

# Notes
- No learning (fixed strategy)
- Useful for benchmarking RL agents
- Simple, interpretable behavior

# Example
```julia
agent = create_baseline_agent(1, 10000.0, :momentum)
# Momentum trader with 10k starting cash
```
"""
function create_baseline_agent(
    id::Int,
    initial_cash::Float64,
    strategy::Symbol
)::BaselineAgent
    if strategy ∉ [:momentum, :mean_reversion, :random]
        throw(ArgumentError("Invalid strategy: $strategy. Must be :momentum, :mean_reversion, or :random"))
    end
    
    position = Position(initial_cash, 0, initial_cash, 0.0)
    price_history = Float64[]
    
    return BaselineAgent(id, position, strategy, price_history)
end

"""
    observe(agent::BaselineAgent, observation::Observation) -> BeliefState

Process observation and update price history.

# Arguments
- `agent::BaselineAgent`: Agent (modified in-place)
- `observation::Observation`: Market observation

# Returns
- `BeliefState`: Dummy belief (baseline doesn't use beliefs)

# Side Effects
- Appends observed price to price_history
- Keeps last 20 prices (sliding window)

# Notes
- Price history used for momentum/mean reversion
- Baseline agents don't maintain full belief states

# Example
```julia
obs = Observation(102.0, 100.0, 1)
belief = observe(agent, obs)
# agent.price_history updated
```
"""
function observe(agent::BaselineAgent, observation::Observation)::BeliefState
    # Update price history
    push!(agent.price_history, observation.observed_price)
    
    # Keep last 20 prices (sliding window)
    if length(agent.price_history) > 20
        popfirst!(agent.price_history)
    end
    
    # Return dummy belief (baseline doesn't use beliefs)
    state = POMDPState(observation.observed_price, 0.0, 0.02, observation.time)
    return BeliefState([state], [1.0])
end

"""
    act(agent::BaselineAgent, belief::BeliefState, current_price::Float64) -> Order

Select action based on strategy.

# Arguments
- `agent::BaselineAgent`: Agent
- `belief::BeliefState`: Current belief (not used)
- `current_price::Float64`: Current market price

# Returns
- `Order`: Selected action with quantity

# Strategies

## Momentum
Follows price trends:
```
price_change = current_price - previous_price
if price_change > threshold: buy
if price_change < -threshold: sell
else: hold
```

Intuition: "The trend is your friend"
- Rising prices → Continue rising (buy)
- Falling prices → Continue falling (sell)

## Mean Reversion
Counters price trends:
```
mean_price = average(price_history)
if current_price < mean_price - threshold: buy (undervalued)
if current_price > mean_price + threshold: sell (overvalued)
else: hold
```

Intuition: "Prices revert to mean"
- Low prices → Will rise (buy)
- High prices → Will fall (sell)

## Random
Random actions:
```
action = random_choice([buy, sell, hold])
```

Intuition: Control baseline (no strategy)

# Notes
- Fixed quantity (10 shares)
- Threshold = 1% of price
- Requires ≥2 prices for momentum
- Requires ≥5 prices for mean reversion

# Example
```julia
order = act(agent, belief, 100.0)
# order based on agent.strategy
```
"""
function act(agent::BaselineAgent, belief::BeliefState, current_price::Float64)::Order
    try
        if agent.strategy == :momentum
            return momentum_strategy(agent, current_price)
        elseif agent.strategy == :mean_reversion
            return mean_reversion_strategy(agent, current_price)
        elseif agent.strategy == :random
            return random_strategy(agent)
        else
            @warn "Unknown strategy, defaulting to hold" strategy=agent.strategy
            return Order(:hold, 0)
        end
    catch e
        @error "Action selection failed" exception=(e, catch_backtrace()) agent_id=agent.id strategy=agent.strategy
        return Order(:hold, 0)
    end
end

"""
    momentum_strategy(agent::BaselineAgent, current_price::Float64) -> Order

Momentum trading strategy.

# Algorithm
1. Compute price change: Δp = p_t - p_{t-1}
2. If Δp > threshold: buy (trend up)
3. If Δp < -threshold: sell (trend down)
4. Else: hold (no clear trend)

# Parameters
- threshold = 1% of current price
- quantity = 10 shares

# Notes
- Requires ≥2 prices in history
- Follows trends (buy rising, sell falling)
- Simple momentum indicator

# Example
```julia
order = momentum_strategy(agent, 100.0)
# Buy if price rose >1%, sell if fell >1%
```
"""
function momentum_strategy(agent::BaselineAgent, current_price::Float64)::Order
    if length(agent.price_history) < 2
        return Order(:hold, 0)
    end
    
    # Compute price change
    prev_price = agent.price_history[end-1]
    price_change = current_price - prev_price
    threshold = current_price * 0.01  # 1% threshold
    
    # Follow trend
    if price_change > threshold
        return Order(:buy, 10)
    elseif price_change < -threshold && agent.position.shares >= 10
        return Order(:sell, 10)
    else
        return Order(:hold, 0)
    end
end

"""
    mean_reversion_strategy(agent::BaselineAgent, current_price::Float64) -> Order

Mean reversion trading strategy.

# Algorithm
1. Compute mean price: μ = mean(price_history)
2. If p < μ - threshold: buy (undervalued)
3. If p > μ + threshold: sell (overvalued)
4. Else: hold (fair value)

# Parameters
- threshold = 2% of mean price
- quantity = 10 shares

# Notes
- Requires ≥5 prices in history
- Assumes prices revert to mean
- Contrarian strategy

# Example
```julia
order = mean_reversion_strategy(agent, 100.0)
# Buy if price <98% of mean, sell if >102% of mean
```
"""
function mean_reversion_strategy(agent::BaselineAgent, current_price::Float64)::Order
    if length(agent.price_history) < 5
        return Order(:hold, 0)
    end
    
    # Compute mean price
    mean_price = sum(agent.price_history) / length(agent.price_history)
    threshold = mean_price * 0.02  # 2% threshold
    
    # Counter trend
    if current_price < mean_price - threshold
        return Order(:buy, 10)
    elseif current_price > mean_price + threshold && agent.position.shares >= 10
        return Order(:sell, 10)
    else
        return Order(:hold, 0)
    end
end

"""
    random_strategy(agent::BaselineAgent) -> Order

Random trading strategy (control baseline).

# Algorithm
1. Sample action uniformly: {buy, sell, hold}
2. Fixed quantity: 10 shares

# Notes
- No intelligence (pure random)
- Useful for benchmarking
- Expected return ≈ 0 (random walk)

# Example
```julia
order = random_strategy(agent)
# Random action with equal probability
```
"""
function random_strategy(agent::BaselineAgent)::Order
    action = rand([:buy, :sell, :hold])
    
    if action == :buy
        return Order(:buy, 10)
    elseif action == :sell && agent.position.shares >= 10
        return Order(:sell, 10)
    else
        return Order(:hold, 0)
    end
end

"""
    reset!(agent::BaselineAgent, initial_cash::Float64)

Reset agent state for new episode.

# Arguments
- `agent::BaselineAgent`: Agent to reset (modified in-place)
- `initial_cash::Float64`: Starting cash

# Notes
- Resets position
- Clears price history
- Strategy unchanged

# Example
```julia
reset!(agent, 10000.0)
# agent.position reset, price_history cleared
```
"""
function reset!(agent::BaselineAgent, initial_cash::Float64)
    agent.position.cash = initial_cash
    agent.position.shares = 0
    agent.position.portfolio_value = initial_cash
    agent.position.pnl = 0.0
    empty!(agent.price_history)
end

"""
    update_position!(agent::BaselineAgent, order::Order, execution_price::Float64,
                    slippage_cost::Float64, current_price::Float64)

Update agent position after order execution.

# Arguments
- `agent::BaselineAgent`: Agent (modified in-place)
- `order::Order`: Executed order
- `execution_price::Float64`: Execution price
- `slippage_cost::Float64`: Slippage cost (not used by baseline)
- `current_price::Float64`: Current market price

# Side Effects
- Updates position (cash, shares, portfolio value)

# Example
```julia
update_position!(agent, Order(:buy, 10), 101.0, 1.0, 100.0)
```
"""
function update_position!(
    agent::BaselineAgent,
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
export create_baseline_agent, observe, act, reset!, update_position!
export momentum_strategy, mean_reversion_strategy, random_strategy

