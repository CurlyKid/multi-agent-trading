"""
Basic Multi-Agent Trading Simulation

Demonstrates core functionality:
- Market initialization
- Agent creation (Q-learning, Policy Gradient, Baseline)
- Simulation execution
- Visualization

Run with: julia --project=. examples/basic_simulation.jl
"""

using MultiAgentTrading
using Plots

println("=" ^ 60)
println("Multi-Agent RL Trading Simulator - Basic Example")
println("=" ^ 60)

# ============================================================================
# 1. Configure Market
# ============================================================================

println("\n[1/5] Configuring market...")

# Market parameters
params = MarketParams(
    0.0001,      # μ: drift (1 basis point per step)
    0.02,        # σ: volatility (2% per step)
    100.0,       # initial_price: $100
    0.01,        # slippage_factor: 1% price impact
    0.005        # observation_noise: 0.5% noise
)

println("  ✓ Market configured")
println("    - Initial price: \$$(params.initial_price)")
println("    - Volatility: $(params.σ * 100)%")
println("    - Drift: $(params.μ * 10000) bps")

# ============================================================================
# 2. Create Agents
# ============================================================================

println("\n[2/5] Creating agents...")

# Agent 1: Q-learning (discrete actions, experience replay)
agent1 = create_qlearning_agent(
    1,           # id
    10000.0,     # initial_cash
    0.1,         # learning_rate (α)
    0.95,        # discount (γ)
    0.1          # epsilon (exploration)
)
println("  ✓ Agent 1: Q-learning (α=0.1, γ=0.95, ε=0.1)")

# Agent 2: Policy gradient (continuous actions, REINFORCE)
agent2 = create_policy_gradient_agent(
    2,           # id
    10000.0,     # initial_cash
    10,          # n_params (policy network size)
    0.01         # learning_rate
)
println("  ✓ Agent 2: Policy Gradient (α=0.01, 10 params)")

# Agent 3: Baseline momentum (no learning)
agent3 = create_baseline_agent(
    3,           # id
    10000.0,     # initial_cash
    :momentum    # strategy (buy rising, sell falling)
)
println("  ✓ Agent 3: Baseline Momentum (no learning)")

agents = [agent1, agent2, agent3]

# ============================================================================
# 3. Run Simulation
# ============================================================================

println("\n[3/5] Running simulation...")
println("  - Duration: 1000 timesteps")
println("  - Agents: 3 (Q-learning, Policy Gradient, Momentum)")

# Run multi-agent simulation
results = run_simulation(params, agents, 1000)

println("  ✓ Simulation complete")
println("    - Final price: \$$(round(results[:price_history][end], digits=2))")
println("    - Price change: $(round((results[:price_history][end] / results[:price_history][1] - 1) * 100, digits=2))%")

# ============================================================================
# 4. Display Results
# ============================================================================

println("\n[4/5] Performance metrics:")
println()

for agent_id in [1, 2, 3]
    metrics = results[:metrics][agent_id]
    
    println("  Agent $agent_id:")
    println("    - Return: $(round(metrics.cumulative_return * 100, digits=2))%")
    println("    - Sharpe: $(round(metrics.sharpe_ratio, digits=2))")
    println("    - Max Drawdown: $(round(metrics.max_drawdown * 100, digits=2))%")
    println("    - Win Rate: $(round(metrics.win_rate * 100, digits=2))%")
    println("    - Trades: $(metrics.total_trades)")
    println()
end

# ============================================================================
# 5. Visualize
# ============================================================================

println("[5/5] Creating visualizations...")

# Comprehensive plot (3 panels)
p = plot_results(results)
savefig(p, "basic_simulation_results.png")
println("  ✓ Saved: basic_simulation_results.png")

# Individual plots
p1 = plot_price_dynamics(results[:price_history])
savefig(p1, "price_dynamics.png")
println("  ✓ Saved: price_dynamics.png")

p2 = plot_agent_positions(results[:agent_histories], [1, 2, 3])
savefig(p2, "agent_positions.png")
println("  ✓ Saved: agent_positions.png")

p3 = plot_performance_metrics(results[:metrics])
savefig(p3, "performance_metrics.png")
println("  ✓ Saved: performance_metrics.png")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 60)
println("Simulation Complete!")
println("=" ^ 60)
println()
println("Key Takeaways:")
println("  • Multi-agent coordination: 3 agents trading simultaneously")
println("  • POMDP framework: Agents observe noisy prices, maintain beliefs")
println("  • RL algorithms: Q-learning and policy gradient learning online")
println("  • Emergent dynamics: Market impact from agent interactions")
println()
println("Next Steps:")
println("  • Run examples/strategy_comparison.jl for detailed analysis")
println("  • Experiment with different parameters")
println("  • Add custom agents or strategies")
println()
println("=" ^ 60)

