# Multi-Agent RL Trading Simulator

A Julia-based, multi-agent, reinforcement learning trading simulator demonstrating advanced decision-making under uncertainty, POMDP framework, and emergent market dynamics.

**Portfolio Project** - Showcases ML/AI skills for quant finance, fintech, and AI research roles.

## Features

- **Realistic Market Simulation**: Geometric Brownian motion with slippage and market impact
- **POMDP Framework**: Partial observability, belief states, uncertainty modeling
- **Multi-Agent Coordination**: 3+ agents trading simultaneously with emergent behavior
- **RL Algorithms**: Q-learning (discrete, off-policy) and policy gradients (continuous, on-policy)
- **Baseline Strategies**: Momentum, mean reversion, random (for comparison)
- **Publication-Quality Visualization**: Price dynamics, positions, performance metrics
- **Professional Code Quality**: Comprehensive docs, error handling, clean architecture

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/CurlyKid/multi-agent-trading.git
cd multi-agent-trading

# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Basic Usage

```julia
using MultiAgentTrading

# Configure market
params = MarketParams(
    0.0001,      # drift
    0.02,        # volatility
    100.0,       # initial price
    0.01,        # slippage factor
    0.005        # observation noise
)

# Create agents
agents = [
    create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1),
    create_policy_gradient_agent(2, 10000.0, 10, 0.01),
    create_baseline_agent(3, 10000.0, :momentum)
]

# Run simulation
results = run_simulation(params, agents, 1000)

# Visualize
plot_results(results)
savefig("results.png")
```

### Run Examples

```bash
# Basic simulation (3 agents, 1000 steps)
julia --project=. examples/basic_simulation.jl

# Strategy comparison (5 strategies, multiple episodes)
julia --project=. examples/strategy_comparison.jl
```

**Note:** Examples generate PNG plots in the current directory. Agents may show negative returns initially (untrained). This is expected - the focus is demonstrating the framework, not optimized trading strategies.

## Initial Results (Non-Optimized)

The `results/` directory contains initial test runs with **untrained agents** (demonstration only):

- `price_dynamics.png` - Market price evolution (GBM)
- `agent_positions.png` - Agent share holdings over time
- `performance_metrics.png` - Cumulative returns (negative = untrained)
- `basic_simulation_results.png` - Full simulation dashboard

**These results show the framework working correctly, not optimal trading.** Negative returns are expected for untrained agents. See "Training and Optimization" section below for achieving profitable strategies.

### Important: Training and Optimization

**This project demonstrates the framework, not optimal trading performance.** Achieving profitable strategies requires:

**1. Extended Training**
- Current: 1,000 timesteps (demonstration)
- Recommended: 10,000-100,000 timesteps (convergence)
- Use `train_agent!` function with higher episode counts

**2. Hyperparameter Optimization**
- **Q-Learning:** Learning rate (α), discount factor (γ), exploration (ε), decay rate
- **Policy Gradient:** Learning rate, policy network size, baseline method
- **Methods:** Bayesian optimization, Hyperband, Grid search
- **Tools:** Optuna.jl, Hyperopt.jl, or custom search

**3. Reward Shaping**
- Current: Simple Δ portfolio value - slippage
- Consider: Risk-adjusted returns (Sharpe), transaction costs, position limits, drawdown penalties

**4. Market Adaptation**
- Different market conditions require different strategies
- Train on historical data, validate on out-of-sample periods
- Consider regime detection (trending vs mean-reverting markets)

**5. Advanced Techniques**
- Deep Q-Networks (DQN) for larger state spaces
- Actor-Critic methods (A2C, PPO) for stability
- Curriculum learning (easy → hard markets)
- Multi-task learning (multiple market conditions)

**Example optimization workflow:**
```julia
# Hyperparameter search (pseudo-code)
using Optuna

function objective(trial)
    α = trial.suggest_float("alpha", 0.001, 0.1)
    γ = trial.suggest_float("gamma", 0.9, 0.99)
    ε = trial.suggest_float("epsilon", 0.01, 0.3)
    
    agent = create_qlearning_agent(1, 10000.0, α, γ, ε)
    stats = train_agent!(agent, params, 100, 1000)
    
    return mean(stats[:returns])  # Maximize average return
end

study = create_study(direction="maximize")
optimize!(study, objective, n_trials=100)
```

**The framework is production-ready. The strategies need tuning for your specific use case.**

## Architecture

**Layered Design** (optimized for clarity and speed):

```
Application Layer (Examples, Experiments)
    ↓
Simulation Layer (Multi-agent coordination)
    ↓
Agent Layer (Q-learning, Policy Gradient, Baselines)
    ↓
POMDP Layer (Belief states, observations)
    ↓
Market Layer (Price dynamics, order execution)
    ↓
Julia Ecosystem (POMDPs.jl, Plots.jl, Distributions.jl)
```

## Components

### Market Environment

- **Geometric Brownian Motion**: `dS = μS dt + σS dW`
- **Slippage**: Price impact proportional to order size
- **Partial Observability**: Noisy price observations

### Agents

**Q-Learning Agent**
- Discrete action space (buy/sell/hold)
- Experience replay for sample efficiency
- ε-greedy exploration

**Policy Gradient Agent**
- Continuous action space (order size)
- REINFORCE algorithm with baseline
- Variance reduction techniques

**Baseline Agents**
- Momentum: Buy rising, sell falling
- Mean reversion: Buy low, sell high
- Random: Control baseline

### POMDP Framework

- **Belief States**: Probability distributions over market states
- **Bayes' Rule**: Belief updates from observations
- **Uncertainty Modeling**: Agents reason about hidden state

### Performance Metrics

- **Cumulative Return**: Total profit/loss (%)
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Maximum Drawdown**: Worst peak-to-trough decline (%)
- **Win Rate**: Fraction of profitable trades
- **Total Trades**: Number of buy/sell actions

## Examples

### Basic Simulation

Demonstrates core functionality with 3 agents trading for 1000 timesteps.

**Output:**
- `basic_simulation_results.png`: Comprehensive 3-panel visualization
- `price_dynamics.png`: Market price evolution
- `agent_positions.png`: Portfolio values over time
- `performance_metrics.png`: Bar chart comparison

### Strategy Comparison

Comprehensive analysis comparing 5 strategies across multiple episodes.

**Output:**
- `strategy_comparison_full.png`: Detailed results
- `return_distribution.png`: Box plot of returns
- `sharpe_comparison.png`: Risk-adjusted performance

## API Documentation

### Market Functions

```julia
initialize_market(params::MarketParams) -> MarketState
update_price(state::MarketState, params::MarketParams) -> Float64
execute_order(state::MarketState, order::Order, params::MarketParams) -> Tuple
observe_market(state::MarketState, params::MarketParams) -> Observation
```

### Agent Functions

```julia
create_qlearning_agent(id, cash, α, γ, ε) -> QLearningAgent
create_policy_gradient_agent(id, cash, n_params, α) -> PolicyGradientAgent
create_baseline_agent(id, cash, strategy) -> BaselineAgent

observe(agent, observation) -> BeliefState
act(agent, belief, price) -> Order
learn!(agent, experience) -> Nothing
reset!(agent, cash) -> Nothing
```

### Simulation Functions

```julia
run_simulation(params, agents, n_steps) -> Dict
train_agent!(agent, params, n_episodes, n_steps) -> Dict
compute_metrics(agent_history, market_history) -> PerformanceMetrics
```

### Visualization Functions

```julia
plot_price_dynamics(price_history) -> Plot
plot_agent_positions(agent_histories, agent_ids) -> Plot
plot_performance_metrics(metrics) -> Plot
plot_results(results) -> Plot
plot_learning_curves(training_stats) -> Plot
animate_trading(results; fps=10) -> Animation
```

## Performance

**Hardware Requirements:**
- CPU: Intel i7-9750H (6 cores) or equivalent
- RAM: 16GB
- Storage: 100MB

**Simulation Benchmarks:**
- 1000 timesteps with 3 agents: < 5 seconds
- 1000 timesteps with 5 agents: < 8 seconds
- 10,000 timesteps with 3 agents: < 30 seconds

**Training Benchmarks (estimated):**
- Q-Learning convergence: 10,000-50,000 timesteps
- Policy Gradient convergence: 50,000-100,000 timesteps
- Hyperparameter search: 100-1000 trials (hours to days)

**Note:** Performance metrics (returns, Sharpe ratio) depend heavily on hyperparameter tuning and training duration. The framework is optimized for speed; the strategies require optimization for profitability.

## Project Structure

```
multi-agent-trading/
├── src/
│   ├── MultiAgentTrading.jl    # Main module
│   ├── types.jl                 # Core type definitions
│   ├── market.jl                # Market simulation
│   ├── pomdp.jl                 # POMDP framework
│   ├── agents/
│   │   ├── qlearning.jl         # Q-learning agent
│   │   ├── policy_gradient.jl   # Policy gradient agent
│   │   └── baseline.jl          # Baseline strategies
│   ├── training.jl              # RL training loop
│   ├── simulation.jl            # Multi-agent simulation
│   └── visualization.jl         # Plotting functions
├── examples/
│   ├── basic_simulation.jl      # Quick start example
│   └── strategy_comparison.jl   # Comprehensive analysis
├── test/                        # Unit and integration tests
├── Project.toml                 # Dependencies
└── README.md                    # This file
```

## Dependencies

- **Julia**: 1.9+
- **POMDPs.jl**: POMDP framework
- **Plots.jl**: Visualization
- **Distributions.jl**: Probability distributions
- **Random.jl**: Random number generation
- **Statistics.jl**: Statistical functions

## Development

### Running Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### Adding Custom Agents

Implement the `TradingAgent` interface:

```julia
mutable struct MyAgent <: TradingAgent
    id::Int
    position::Position
    # ... custom fields
end

function observe(agent::MyAgent, observation::Observation)
    # Update belief
end

function act(agent::MyAgent, belief::BeliefState, price::Float64)
    # Select action
end

function learn!(agent::MyAgent, experience)
    # Update parameters
end

function reset!(agent::MyAgent, cash::Float64)
    # Reset state
end
```

## Future Enhancements

**Framework Extensions:**
- [ ] Deep Q-Networks (DQN) - Neural network function approximation
- [ ] Actor-Critic methods (A2C, PPO) - Improved stability
- [ ] Multi-asset trading - Portfolio optimization
- [ ] Order book simulation - Market microstructure
- [ ] Real market data integration - Historical backtesting

**Optimization and Deployment:**
- [ ] Hyperparameter optimization (Bayesian, Hyperband)
- [ ] Distributed training (multi-core, multi-node)
- [ ] Model checkpointing and versioning
- [ ] Live trading interface (paper trading first!)
- [ ] Risk management system (position limits, stop-loss)

## License

MIT License - See LICENSE file for details

## Author

**Avery Watts**
- GitHub: [@CurlyKid](https://github.com/CurlyKid)
- Email: nagarake@yahoo.com

## Acknowledgments

- Built with Julia for high-performance numerical computing
- POMDP framework inspired by POMDPs.jl ecosystem
- Reinforcement learning algorithms based on Sutton & Barto (2018)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multi_agent_trading,
  author = {Avery Watts},
  title = {Multi-Agent RL Trading Simulator},
  year = {2026},
  url = {https://github.com/CurlyKid/multi-agent-trading}
}
```

---

**Portfolio Project** - Demonstrates adequate skills in:
- Reinforcement Learning (Q-learning, Policy Gradients)
- POMDP Framework (Partial Observability, Belief States)
- Multi-Agent Systems (Coordination, Emergent Behavior)
- Financial ML (Trading, Risk Management)
- Software Engineering (Clean Code, Documentation, Testing)

