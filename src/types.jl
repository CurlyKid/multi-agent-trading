"""
Core type definitions for Multi-Agent RL Trading Simulator.

This module defines all fundamental types used throughout the system:
- Market state and parameters
- Agent types and positions
- POMDP framework types
- Actions, observations, and experiences
- Performance metrics
"""

using Distributions

# ============================================================================
# Market Types
# ============================================================================

"""
    MarketParams

Configuration parameters for market simulation.

# Fields
- `μ::Float64`: Drift (expected return per timestep)
- `σ::Float64`: Volatility (standard deviation of returns)
- `initial_price::Float64`: Starting market price
- `slippage_factor::Float64`: Price impact coefficient (0 = no slippage)
- `observation_noise::Float64`: Noise level in price observations (0 = perfect info)

# Example
```julia
params = MarketParams(
    μ = 0.0001,              # 1 basis point drift
    σ = 0.02,                # 2% volatility
    initial_price = 100.0,
    slippage_factor = 0.01,  # 1% slippage per unit
    observation_noise = 0.005 # 0.5% observation noise
)
```

# Notes
- Drift μ represents expected price change per timestep
- Volatility σ controls price fluctuation magnitude
- Slippage increases execution cost for large orders
- Observation noise creates partial observability (POMDP)
"""
struct MarketParams
    μ::Float64
    σ::Float64
    initial_price::Float64
    slippage_factor::Float64
    observation_noise::Float64
end

"""
    MarketState

Current state of the market simulation.

# Fields
- `time::Int`: Current timestep
- `price::Float64`: Current market price
- `volume::Float64`: Total trading volume this timestep
- `price_history::Vector{Float64}`: Historical prices
- `volume_history::Vector{Float64}`: Historical volumes

# Invariants
- `price > 0` (prices must be positive)
- `length(price_history) == time + 1` (includes initial price)
- `length(volume_history) == time + 1`
"""
mutable struct MarketState
    time::Int
    price::Float64
    volume::Float64
    price_history::Vector{Float64}
    volume_history::Vector{Float64}
end

# ============================================================================
# Agent Types
# ============================================================================

"""
    TradingAgent

Abstract base type for all trading agents.

All concrete agent types must implement:
- `observe(agent, observation)`: Process market observation
- `act(agent, belief_state)`: Select action based on belief
- `learn!(agent, experience)`: Update from experience
- `reset!(agent)`: Reset agent state

# Design Pattern
Uses Julia's multiple dispatch for strategy polymorphism.
Easy to add new agent types by implementing the interface.
"""
abstract type TradingAgent end

"""
    Position

Agent's current position and portfolio state.

# Fields
- `cash::Float64`: Available cash
- `shares::Int`: Number of shares held
- `portfolio_value::Float64`: Total value (cash + shares * price)
- `pnl::Float64`: Profit/loss since start

# Invariants
- `portfolio_value = cash + shares * current_price`
- `pnl = portfolio_value - initial_portfolio_value`
"""
mutable struct Position
    cash::Float64
    shares::Int
    portfolio_value::Float64
    pnl::Float64
end

"""
    QLearningAgent <: TradingAgent

Q-learning agent for discrete action spaces.

# Fields
- `id::Int`: Unique agent identifier
- `position::Position`: Current position and portfolio
- `q_table::Dict{Tuple{Int,Symbol}, Float64}`: State-action value estimates
- `learning_rate::Float64`: α parameter (0 < α ≤ 1)
- `discount::Float64`: γ parameter (0 ≤ γ < 1)
- `epsilon::Float64`: Exploration rate (0 ≤ ε ≤ 1)
- `experience_buffer::Vector{Experience}`: Replay buffer

# Algorithm
Q-learning update rule:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

# Action Selection
ε-greedy: random action with probability ε, otherwise argmax_a Q(s,a)

# Notes
- Experience replay improves sample efficiency
- Q-values bounded to prevent divergence
- Discrete state space (price discretized into bins)
"""
mutable struct QLearningAgent <: TradingAgent
    id::Int
    position::Position
    q_table::Dict{Tuple{Int,Symbol}, Float64}
    learning_rate::Float64
    discount::Float64
    epsilon::Float64
    experience_buffer::Vector{Any}  # Will hold Experience tuples
end

"""
    PolicyGradientAgent <: TradingAgent

Policy gradient agent using REINFORCE algorithm.

# Fields
- `id::Int`: Unique agent identifier
- `position::Position`: Current position and portfolio
- `policy_params::Vector{Float64}`: Policy network parameters
- `baseline::Float64`: Value function baseline (reduces variance)
- `learning_rate::Float64`: Step size for gradient updates
- `trajectory::Vector{Tuple}`: Current episode trajectory

# Algorithm
REINFORCE policy gradient:
```
∇J(θ) = E[∇log π(a|s,θ) * (G - b)]
```
where G is discounted return, b is baseline

# Notes
- Baseline reduces variance without introducing bias
- Gradients clipped for stability
- Continuous action space (order size)
"""
mutable struct PolicyGradientAgent <: TradingAgent
    id::Int
    position::Position
    policy_params::Vector{Float64}
    baseline::Float64
    learning_rate::Float64
    trajectory::Vector{Tuple}
end

"""
    BaselineAgent <: TradingAgent

Simple heuristic agent (no learning).

# Fields
- `id::Int`: Unique agent identifier
- `position::Position`: Current position and portfolio
- `strategy::Symbol`: Strategy type (:momentum, :mean_reversion, :random)
- `price_history::Vector{Float64}`: Recent prices for strategy

# Strategies
- `:momentum`: Buy if price increasing, sell if decreasing
- `:mean_reversion`: Buy if price below average, sell if above
- `:random`: Random actions (baseline for comparison)

# Notes
- No learning (fixed strategy)
- Useful for benchmarking learned strategies
- Deterministic (except :random)
"""
mutable struct BaselineAgent <: TradingAgent
    id::Int
    position::Position
    strategy::Symbol
    price_history::Vector{Float64}
end

# ============================================================================
# POMDP Types
# ============================================================================

"""
    POMDPState

True market state (not directly observable).

# Fields
- `price::Float64`: True market price
- `trend::Float64`: Current price trend
- `volatility::Float64`: Current volatility
- `time::Int`: Timestep

# Notes
- Agents don't observe this directly (partial observability)
- Agents maintain belief distribution over possible states
- Trend and volatility may change over time
"""
struct POMDPState
    price::Float64
    trend::Float64
    volatility::Float64
    time::Int
end

"""
    Observation

Noisy/partial observation of market state.

# Fields
- `observed_price::Float64`: Noisy price observation
- `volume::Float64`: Trading volume (observable)
- `time::Int`: Timestep

# Notes
- observed_price = true_price * (1 + noise)
- noise ~ N(0, observation_noise)
- Creates partial observability for POMDP framework
"""
struct Observation
    observed_price::Float64
    volume::Float64
    time::Int
end

"""
    BeliefState

Probability distribution over possible states.

# Fields
- `states::Vector{POMDPState}`: Possible states
- `weights::Vector{Float64}`: Probabilities (sum to 1)

# Invariants
- All weights non-negative
- sum(weights) ≈ 1.0 (within numerical tolerance)
- length(states) == length(weights)

# Notes
- Updated using Bayes' rule: P(s|o) ∝ P(o|s) * P(s)
- Represents agent's uncertainty about true state
"""
struct BeliefState
    states::Vector{POMDPState}
    weights::Vector{Float64}
end

# ============================================================================
# Action and Experience Types
# ============================================================================

"""
    Order

Trading action.

# Fields
- `action::Symbol`: Action type (:buy, :sell, :hold)
- `quantity::Int`: Number of shares (0 for :hold)

# Notes
- Positive quantity for buy/sell
- Zero quantity for hold
- Executed with slippage based on quantity
"""
struct Order
    action::Symbol
    quantity::Int
end

"""
    Experience

Single experience tuple for reinforcement learning.

# Fields
- `state::Int`: Current state (discretized)
- `action::Symbol`: Action taken
- `reward::Float64`: Reward received
- `next_state::Int`: Next state
- `done::Bool`: Episode terminal flag

# Notes
- Used for Q-learning updates
- Stored in experience replay buffer
- Enables off-policy learning
"""
struct Experience
    state::Int
    action::Symbol
    reward::Float64
    next_state::Int
    done::Bool
end

# ============================================================================
# Performance Metrics
# ============================================================================

"""
    PerformanceMetrics

Agent performance statistics.

# Fields
- `cumulative_return::Float64`: Total return (%)
- `sharpe_ratio::Float64`: Risk-adjusted return
- `max_drawdown::Float64`: Maximum peak-to-trough decline (%)
- `win_rate::Float64`: Fraction of profitable trades
- `total_trades::Int`: Number of trades executed

# Formulas
- cumulative_return = (final_value - initial_value) / initial_value
- sharpe_ratio = mean(returns) / std(returns)
- max_drawdown = max((peak - trough) / peak)
- win_rate = profitable_trades / total_trades

# Notes
- Sharpe ratio measures risk-adjusted performance
- Max drawdown indicates worst-case loss
- Win rate alone can be misleading (need return magnitude)
"""
struct PerformanceMetrics
    cumulative_return::Float64
    sharpe_ratio::Float64
    max_drawdown::Float64
    win_rate::Float64
    total_trades::Int
end

# ============================================================================
# Exports
# ============================================================================

export MarketParams, MarketState
export TradingAgent, Position
export QLearningAgent, PolicyGradientAgent, BaselineAgent
export POMDPState, Observation, BeliefState
export Order, Experience
export PerformanceMetrics
