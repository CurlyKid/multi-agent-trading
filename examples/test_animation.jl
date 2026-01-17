"""
Test Animation

Quick test of animate_trading function.
Creates short animation (100 timesteps) to verify functionality.

Run with: julia --project=. examples/test_animation.jl
"""

using MultiAgentTrading
using Plots

println("=" ^ 60)
println("Testing Animation")
println("=" ^ 60)

# Configure market
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)

# Create agents
agents = [
    create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1),
    create_policy_gradient_agent(2, 10000.0, 10, 0.01),
    create_baseline_agent(3, 10000.0, :momentum)
]

println("\n[1/3] Running simulation (100 timesteps)...")
results = run_simulation(params, agents, 100)

println("\n[2/3] Creating animation...")
anim = animate_trading(results, fps=10, filename="trading_animation.gif")

println("\n[3/3] Animation complete!")
println("  ✓ Saved: trading_animation.gif")
println("  ✓ Frames: 100")
println("  ✓ FPS: 10")
println("  ✓ Duration: ~10 seconds")

println("\n" * "=" ^ 60)
println("Animation test complete!")
println("=" ^ 60)
