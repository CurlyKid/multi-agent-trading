"""
    MultiAgentTrading

Multi-agent reinforcement learning trading simulator.

Demonstrates POMDP expertise, multi-agent coordination, and RL implementation
for portfolio presentation to quant finance, fintech, and AI research roles.

# Features
- Realistic market simulation (geometric Brownian motion)
- POMDP framework (partial observability, belief states)
- Multiple RL algorithms (Q-learning, policy gradients)
- Multi-agent coordination (emergent market dynamics)
- Publication-quality visualization

# Quick Start
```julia
using MultiAgentTrading

# Create market
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
market = initialize_market(params)

# Create agents
agents = [
    QLearningAgent(1, ...),
    PolicyGradientAgent(2, ...),
    BaselineAgent(3, ..., :momentum)
]

# Run simulation
results = run_simulation(market, agents, params, n_steps=1000)

# Visualize
plot_results(results)
```

# Architecture
Layered architecture with clear separation:
- Market layer: Price dynamics, order execution
- Agent layer: Trading strategies, learning algorithms
- POMDP layer: Belief state management
- Simulation layer: Multi-agent coordination
- Visualization layer: Plots and metrics

# Author
Portfolio project demonstrating senior ML/AI skills

# Version
0.1.0
"""
module MultiAgentTrading

# Core types
include("types.jl")

# Market environment
include("market.jl")

# POMDP framework
include("pomdp.jl")

# Agent implementations
include("agents/qlearning.jl")
include("agents/policy_gradient.jl")
include("agents/baseline.jl")

# Training loop
include("training.jl")

# Multi-agent simulation
include("simulation.jl")

# Visualization
include("visualization.jl")

end # module
