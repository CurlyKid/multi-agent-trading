"""
Strategy Comparison: Q-Learning vs Policy Gradient vs Baselines

Comprehensive analysis comparing:
- Q-learning (discrete, off-policy)
- Policy gradient (continuous, on-policy)
- Momentum baseline (trend following)
- Mean reversion baseline (contrarian)
- Random baseline (control)

Run with: julia --project=. examples/strategy_comparison.jl
"""

using MultiAgentTrading
using Plots
using Statistics

println("=" ^ 70)
println("Multi-Agent RL Trading Simulator - Strategy Comparison")
println("=" ^ 70)

# ============================================================================
# Configuration
# ============================================================================

println("\n[Configuration]")

# Market parameters
params = MarketParams(
    0.0001,      # μ: slight upward drift
    0.02,        # σ: 2% volatility
    100.0,       # initial_price
    0.01,        # slippage_factor
    0.005        # observation_noise
)

# Simulation settings
n_episodes = 5          # Number of independent runs
n_steps = 1000          # Steps per episode
initial_cash = 10000.0  # Starting capital

println("  Market: μ=$(params.μ), σ=$(params.σ), P₀=\$$(params.initial_price)")
println("  Simulation: $(n_episodes) episodes × $(n_steps) steps")
println("  Initial capital: \$$(initial_cash)")

# ============================================================================
# Create Agents
# ============================================================================

println("\n[Creating Agents]")

function create_agent_set(episode::Int)
    return [
        create_qlearning_agent(1, initial_cash, 0.1, 0.95, 0.1),
        create_policy_gradient_agent(2, initial_cash, 10, 0.01),
        create_baseline_agent(3, initial_cash, :momentum),
        create_baseline_agent(4, initial_cash, :mean_reversion),
        create_baseline_agent(5, initial_cash, :random)
    ]
end

agent_names = Dict(
    1 => "Q-Learning",
    2 => "Policy Gradient",
    3 => "Momentum",
    4 => "Mean Reversion",
    5 => "Random"
)

println("  ✓ 5 strategies configured")
for (id, name) in sort(agent_names)
    println("    Agent $id: $name")
end

# ============================================================================
# Run Multiple Episodes
# ============================================================================

println("\n[Running $(n_episodes) Episodes]")

all_results = []
all_metrics = Dict(id => [] for id in 1:5)

for episode in 1:n_episodes
    print("  Episode $episode/$n_episodes... ")
    
    # Create fresh agents
    agents = create_agent_set(episode)
    
    # Run simulation
    results = run_simulation(params, agents, n_steps)
    push!(all_results, results)
    
    # Collect metrics
    for agent_id in 1:5
        push!(all_metrics[agent_id], results[:metrics][agent_id])
    end
    
    println("✓")
end

# ============================================================================
# Aggregate Statistics
# ============================================================================

println("\n[Aggregate Performance]")
println()
println("Strategy              | Return (%) | Sharpe  | Drawdown (%) | Win Rate (%) | Trades")
println("-" ^ 90)

for agent_id in 1:5
    metrics_list = all_metrics[agent_id]
    
    # Compute averages
    avg_return = mean([m.cumulative_return for m in metrics_list]) * 100
    avg_sharpe = mean([m.sharpe_ratio for m in metrics_list])
    avg_drawdown = mean([m.max_drawdown for m in metrics_list]) * 100
    avg_win_rate = mean([m.win_rate for m in metrics_list]) * 100
    avg_trades = mean([m.total_trades for m in metrics_list])
    
    # Compute std dev
    std_return = std([m.cumulative_return for m in metrics_list]) * 100
    
    name = agent_names[agent_id]
    println(@sprintf("%-20s | %6.2f±%.2f | %7.2f | %12.2f | %12.2f | %6.1f",
                     name, avg_return, std_return, avg_sharpe, 
                     avg_drawdown, avg_win_rate, avg_trades))
end

println()

# ============================================================================
# Statistical Analysis
# ============================================================================

println("[Statistical Analysis]")
println()

# Best performer by return
returns_by_agent = Dict(
    id => [m.cumulative_return for m in all_metrics[id]]
    for id in 1:5
)

best_agent = argmax(Dict(id => mean(returns) for (id, returns) in returns_by_agent))
println("  Best performer (return): $(agent_names[best_agent])")

# Best risk-adjusted (Sharpe)
sharpes_by_agent = Dict(
    id => [m.sharpe_ratio for m in all_metrics[id]]
    for id in 1:5
)

best_sharpe = argmax(Dict(id => mean(sharpes) for (id, sharpes) in sharpes_by_agent))
println("  Best risk-adjusted (Sharpe): $(agent_names[best_sharpe])")

# Most consistent (lowest std dev of returns)
std_devs = Dict(id => std(returns) for (id, returns) in returns_by_agent)
most_consistent = argmin(std_devs)
println("  Most consistent: $(agent_names[most_consistent])")

# RL vs Baseline comparison
rl_returns = vcat(returns_by_agent[1], returns_by_agent[2])
baseline_returns = vcat(returns_by_agent[3], returns_by_agent[4], returns_by_agent[5])

println()
println("  RL agents (Q-learning + Policy Gradient):")
println("    Mean return: $(round(mean(rl_returns) * 100, digits=2))%")
println("    Std dev: $(round(std(rl_returns) * 100, digits=2))%")
println()
println("  Baseline agents (Momentum + Mean Reversion + Random):")
println("    Mean return: $(round(mean(baseline_returns) * 100, digits=2))%")
println("    Std dev: $(round(std(baseline_returns) * 100, digits=2))%")

# ============================================================================
# Visualizations
# ============================================================================

println("\n[Creating Visualizations]")

# Use last episode for detailed plots
last_results = all_results[end]

# 1. Comprehensive results
p_full = plot_results(last_results)
savefig(p_full, "strategy_comparison_full.png")
println("  ✓ Saved: strategy_comparison_full.png")

# 2. Return distribution (box plot)
p_returns = plot(
    xlabel="Strategy",
    ylabel="Return (%)",
    title="Return Distribution Across Episodes",
    legend=false,
    size=(800, 500)
)

for agent_id in 1:5
    returns = [m.cumulative_return * 100 for m in all_metrics[agent_id]]
    boxplot!(p_returns, [agent_names[agent_id]], returns, 
             fillalpha=0.5, linewidth=2)
end

savefig(p_returns, "return_distribution.png")
println("  ✓ Saved: return_distribution.png")

# 3. Sharpe ratio comparison
p_sharpe = plot(
    xlabel="Strategy",
    ylabel="Sharpe Ratio",
    title="Risk-Adjusted Performance",
    legend=false,
    size=(800, 500)
)

sharpe_means = [mean([m.sharpe_ratio for m in all_metrics[id]]) for id in 1:5]
sharpe_stds = [std([m.sharpe_ratio for m in all_metrics[id]]) for id in 1:5]

bar!(p_sharpe, [agent_names[id] for id in 1:5], sharpe_means,
     yerror=sharpe_stds, fillalpha=0.7, linewidth=2)

savefig(p_sharpe, "sharpe_comparison.png")
println("  ✓ Saved: sharpe_comparison.png")

# 4. Learning curves (if training data available)
# Note: This example runs simulation (not training), so learning curves
# would require separate training runs. Placeholder for future enhancement.

# ============================================================================
# Summary
# ============================================================================

println("\n" * "=" ^ 70)
println("Strategy Comparison Complete!")
println("=" ^ 70)
println()
println("Key Findings:")
println("  • Tested 5 strategies across $(n_episodes) independent episodes")
println("  • Best performer: $(agent_names[best_agent])")
println("  • Best risk-adjusted: $(agent_names[best_sharpe])")
println("  • Most consistent: $(agent_names[most_consistent])")
println()
println("Insights:")
println("  • RL agents learn from experience (improve over time)")
println("  • Momentum works in trending markets")
println("  • Mean reversion works in ranging markets")
println("  • Random baseline confirms skill vs luck")
println("  • Multi-agent interactions create emergent dynamics")
println()
println("Files Generated:")
println("  • strategy_comparison_full.png (comprehensive results)")
println("  • return_distribution.png (box plot)")
println("  • sharpe_comparison.png (risk-adjusted performance)")
println()
println("=" ^ 70)

