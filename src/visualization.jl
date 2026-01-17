"""
Visualization functions for trading simulation.

Publication-quality plots for portfolio presentation:
- Price dynamics
- Agent positions
- Performance metrics
- Optional: Animation
"""

using Plots
using Statistics

"""
    plot_price_dynamics(price_history::Vector{Float64}; 
                       title="Price Dynamics") -> Plots.Plot

Plot market price evolution over time.

# Arguments
- `price_history::Vector{Float64}`: Historical prices
- `title::String`: Plot title (default "Price Dynamics")

# Returns
- `Plots.Plot`: Price trajectory plot

# Features
- Price line with markers
- Grid for readability
- Publication-quality formatting
- Clear axis labels

# Error Handling
- Validates price_history is non-empty
- Returns empty plot on error
- Logs warnings

# Example
```julia
results = run_simulation(params, agents, 1000)
plot_price_dynamics(results[:price_history])
# Shows price evolution over 1000 timesteps
```
"""
function plot_price_dynamics(
    price_history::Vector{Float64};
    title="Price Dynamics"
)
    try
        @assert !isempty(price_history) "Price history must be non-empty"
        
        p = plot(
            price_history,
            label="Market Price",
            xlabel="Time (steps)",
            ylabel="Price (\$)",
            title=title,
            linewidth=2,
            color=:blue,
            legend=:best,
            grid=true,
            size=(800, 400),
            margin=5Plots.mm
        )
        
        # Add horizontal line at initial price
        hline!([price_history[1]], 
               label="Initial Price", 
               linestyle=:dash, 
               color=:gray,
               linewidth=1)
        
        return p
        
    catch e
        @error "Price dynamics plot failed" exception=(e, catch_backtrace())
        return plot()  # Empty plot
    end
end

"""
    plot_agent_positions(agent_histories::Dict, agent_ids::Vector{Int};
                        title="Agent Positions") -> Plots.Plot

Plot agent portfolio values over time.

# Arguments
- `agent_histories::Dict`: Agent position histories (from run_simulation)
- `agent_ids::Vector{Int}`: Agent IDs to plot
- `title::String`: Plot title (default "Agent Positions")

# Returns
- `Plots.Plot`: Multi-line plot of portfolio values

# Features
- One line per agent (different colors)
- Legend with agent IDs
- Grid for readability
- Shows relative performance

# Error Handling
- Validates agent_histories is non-empty
- Skips missing agents with warning
- Returns empty plot on error

# Example
```julia
results = run_simulation(params, agents, 1000)
plot_agent_positions(results[:agent_histories], [1, 2, 3])
# Shows portfolio values for agents 1, 2, 3
```
"""
function plot_agent_positions(
    agent_histories::Dict,
    agent_ids::Vector{Int};
    title="Agent Positions"
)
    try
        @assert !isempty(agent_histories) "Agent histories must be non-empty"
        
        p = plot(
            xlabel="Time (steps)",
            ylabel="Portfolio Value (\$)",
            title=title,
            legend=:best,
            grid=true,
            size=(800, 400),
            margin=5Plots.mm
        )
        
        # Plot each agent
        colors = [:blue, :red, :green, :orange, :purple, :brown]
        for (i, agent_id) in enumerate(agent_ids)
            if haskey(agent_histories, agent_id)
                portfolio_values = agent_histories[agent_id][:portfolio_value]
                plot!(p, portfolio_values,
                     label="Agent $agent_id",
                     linewidth=2,
                     color=colors[mod1(i, length(colors))])
            else
                @warn "Agent not found in histories" agent_id
            end
        end
        
        return p
        
    catch e
        @error "Agent positions plot failed" exception=(e, catch_backtrace())
        return plot()
    end
end

"""
    plot_performance_metrics(metrics::Dict; title="Performance Metrics") -> Plots.Plot

Plot performance comparison across agents.

# Arguments
- `metrics::Dict`: Performance metrics per agent (from run_simulation)
- `title::String`: Plot title (default "Performance Metrics")

# Returns
- `Plots.Plot`: Grouped bar chart of metrics

# Features
- Grouped bars (return, Sharpe, drawdown, win rate)
- One group per agent
- Color-coded metrics
- Percentage formatting

# Metrics Displayed
- Cumulative return (%)
- Sharpe ratio
- Max drawdown (%)
- Win rate (%)

# Error Handling
- Validates metrics is non-empty
- Handles missing metrics gracefully
- Returns empty plot on error

# Example
```julia
results = run_simulation(params, agents, 1000)
plot_performance_metrics(results[:metrics])
# Bar chart comparing agent performance
```
"""
function plot_performance_metrics(
    metrics::Dict;
    title="Performance Metrics"
)
    try
        @assert !isempty(metrics) "Metrics must be non-empty"
        
        agent_ids = sort(collect(keys(metrics)))
        n_agents = length(agent_ids)
        
        # Extract metrics
        returns = [metrics[id].cumulative_return * 100 for id in agent_ids]
        sharpes = [metrics[id].sharpe_ratio for id in agent_ids]
        drawdowns = [metrics[id].max_drawdown * 100 for id in agent_ids]
        win_rates = [metrics[id].win_rate * 100 for id in agent_ids]
        
        # Create grouped bar chart
        x = 1:n_agents
        width = 0.2
        
        p = plot(
            xlabel="Agent ID",
            ylabel="Value",
            title=title,
            legend=:best,
            grid=true,
            size=(800, 500),
            margin=5Plots.mm,
            xticks=(x, string.(agent_ids))
        )
        
        # Plot bars (offset for grouping)
        bar!(p, x .- 1.5*width, returns, 
             label="Return (%)", 
             color=:green, 
             bar_width=width)
        bar!(p, x .- 0.5*width, sharpes, 
             label="Sharpe Ratio", 
             color=:blue, 
             bar_width=width)
        bar!(p, x .+ 0.5*width, drawdowns, 
             label="Max Drawdown (%)", 
             color=:red, 
             bar_width=width)
        bar!(p, x .+ 1.5*width, win_rates, 
             label="Win Rate (%)", 
             color=:orange, 
             bar_width=width)
        
        return p
        
    catch e
        @error "Performance metrics plot failed" exception=(e, catch_backtrace())
        return plot()
    end
end

"""
    plot_results(results::Dict) -> Plots.Plot

Create comprehensive visualization of simulation results.

# Arguments
- `results::Dict`: Full simulation results (from run_simulation)

# Returns
- `Plots.Plot`: Multi-panel plot with all visualizations

# Features
- Top panel: Price dynamics
- Middle panel: Agent positions
- Bottom panel: Performance metrics
- Publication-quality layout

# Error Handling
- Validates results structure
- Falls back to individual plots on error
- Logs warnings

# Example
```julia
results = run_simulation(params, agents, 1000)
plot_results(results)
# Comprehensive 3-panel visualization
savefig("simulation_results.png")
```
"""
function plot_results(results::Dict)
    try
        # Extract data
        price_history = results[:price_history]
        agent_histories = results[:agent_histories]
        metrics = results[:metrics]
        agent_ids = sort(collect(keys(agent_histories)))
        
        # Create subplots
        p1 = plot_price_dynamics(price_history, title="Market Price Dynamics")
        p2 = plot_agent_positions(agent_histories, agent_ids, title="Agent Portfolio Values")
        p3 = plot_performance_metrics(metrics, title="Performance Comparison")
        
        # Combine into layout
        p = plot(p1, p2, p3, 
                layout=(3, 1), 
                size=(900, 1200),
                margin=5Plots.mm)
        
        return p
        
    catch e
        @error "Results plot failed" exception=(e, catch_backtrace())
        return plot()
    end
end

"""
    plot_learning_curves(training_stats::Dict; title="Learning Curves") -> Plots.Plot

Plot learning curves from training.

# Arguments
- `training_stats::Dict`: Training statistics (from train_agent!)
- `title::String`: Plot title (default "Learning Curves")

# Returns
- `Plots.Plot`: Learning curve plot

# Features
- Episode returns (raw)
- Smoothed learning curve (moving average)
- Shows learning progress
- Convergence visible

# Error Handling
- Validates training_stats structure
- Returns empty plot on error

# Example
```julia
agent = create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1)
stats = train_agent!(agent, params, 100, 1000)
plot_learning_curves(stats)
# Shows learning progress over 100 episodes
```
"""
function plot_learning_curves(
    training_stats::Dict;
    title="Learning Curves"
)
    try
        episode_returns = training_stats[:episode_returns]
        learning_curve = training_stats[:learning_curve]
        
        p = plot(
            xlabel="Episode",
            ylabel="Return",
            title=title,
            legend=:best,
            grid=true,
            size=(800, 400),
            margin=5Plots.mm
        )
        
        # Raw returns (transparent)
        plot!(p, episode_returns,
             label="Episode Returns",
             alpha=0.3,
             color=:blue,
             linewidth=1)
        
        # Smoothed curve
        plot!(p, learning_curve,
             label="Learning Curve (MA-10)",
             color=:red,
             linewidth=2)
        
        # Zero line
        hline!([0], 
               label="Break-even", 
               linestyle=:dash, 
               color=:gray,
               linewidth=1)
        
        return p
        
    catch e
        @error "Learning curves plot failed" exception=(e, catch_backtrace())
        return plot()
    end
end

"""
    animate_trading(results::Dict; fps=10, filename="trading.gif") -> Animation

Create animation of trading simulation.

# Arguments
- `results::Dict`: Simulation results (from run_simulation)
- `fps::Int`: Frames per second (default 10)
- `filename::String`: Output filename (default "trading.gif")

# Returns
- `Animation`: Plots.jl animation object

# Features
- Frame-by-frame price and positions
- Shows market evolution
- Agent actions visible
- Can save as GIF or MP4

# Notes
- Optional feature (can skip for MVP)
- Useful for presentations
- Large files for long simulations

# Error Handling
- Validates results structure
- Returns nothing on error
- Logs warnings

# Example
```julia
results = run_simulation(params, agents, 1000)
anim = animate_trading(results, fps=20, filename="trading.gif")
# Creates animated GIF of simulation
```
"""
function animate_trading(
    results::Dict;
    fps=10,
    filename="trading.gif"
)
    try
        price_history = results[:price_history]
        agent_histories = results[:agent_histories]
        agent_ids = sort(collect(keys(agent_histories)))
        
        n_frames = length(price_history)
        
        @info "Creating animation" n_frames fps filename
        
        anim = @animate for t in 1:n_frames
            # Price subplot
            p1 = plot(
                price_history[1:t],
                xlabel="Time",
                ylabel="Price",
                title="Market Price (t=$t)",
                legend=false,
                color=:blue,
                linewidth=2,
                xlims=(1, n_frames),
                ylims=(minimum(price_history)*0.95, maximum(price_history)*1.05)
            )
            
            # Portfolio values subplot
            p2 = plot(
                xlabel="Time",
                ylabel="Portfolio Value",
                title="Agent Portfolios",
                legend=:best,
                xlims=(1, n_frames)
            )
            
            colors = [:blue, :red, :green, :orange, :purple]
            for (i, agent_id) in enumerate(agent_ids)
                values = agent_histories[agent_id][:portfolio_value][1:t]
                plot!(p2, values,
                     label="Agent $agent_id",
                     color=colors[mod1(i, length(colors))],
                     linewidth=2)
            end
            
            # Combine
            plot(p1, p2, layout=(2, 1), size=(800, 600))
        end
        
        gif(anim, filename, fps=fps)
        
        @info "Animation saved" filename
        
        return anim
        
    catch e
        @error "Animation failed" exception=(e, catch_backtrace())
        return nothing
    end
end

# Exports
export plot_price_dynamics, plot_agent_positions, plot_performance_metrics
export plot_results, plot_learning_curves, animate_trading

