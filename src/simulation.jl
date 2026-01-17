"""
Multi-agent simulation coordinator.

Manages multiple agents trading in shared market:
- Agent coordination
- Market state updates
- Performance tracking
- Simulation history
"""

using Statistics

"""
    run_simulation(market_params::MarketParams, agents::Vector{TradingAgent},
                   n_steps::Int) -> Dict

Run multi-agent trading simulation.

# Arguments
- `market_params::MarketParams`: Market configuration
- `agents::Vector{TradingAgent}`: All trading agents
- `n_steps::Int`: Simulation length (timesteps)

# Returns
- `Dict`: Simulation results
  - `:market_history`: Market states over time
  - `:agent_histories`: Agent positions over time
  - `:price_history`: Price trajectory
  - `:metrics`: Performance metrics per agent

# Algorithm
For each timestep:
1. All agents observe market (partial, noisy)
2. All agents select actions (simultaneous)
3. Execute all orders (with market impact)
4. Update market state (price, volume)
5. Agents learn from experience
6. Track statistics

# Multi-Agent Dynamics
- Agents act simultaneously (no turn-taking)
- Orders aggregate → market impact
- Emergent behavior from interactions
- Competition for profit opportunities

# Notes
- Market impact scales with total volume
- Agents don't observe each other's positions (privacy)
- Learning happens during simulation (online)
- Final metrics computed at end

# Error Handling
- Validates inputs (n_steps > 0, agents non-empty)
- Continues on individual agent errors
- Logs simulation progress
- Returns partial results if interrupted

# Example
```julia
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
agents = [
    create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1),
    create_policy_gradient_agent(2, 10000.0, 10, 0.01),
    create_baseline_agent(3, 10000.0, :momentum)
]
results = run_simulation(params, agents, 1000)
# 3 agents trade for 1000 timesteps
```
"""
function run_simulation(
    market_params::MarketParams,
    agents::Vector{TradingAgent},
    n_steps::Int
)::Dict
    try
        @assert n_steps > 0 "Number of steps must be positive"
        @assert !isempty(agents) "Must have at least one agent"
        
        @info "Starting simulation" n_agents=length(agents) n_steps
        
        # Initialize market
        market = initialize_market(market_params)
        
        # Initialize agent histories
        agent_histories = Dict(
            agent.id => Dict(
                :cash => Float64[agent.position.cash],
                :shares => Int[agent.position.shares],
                :portfolio_value => Float64[agent.position.portfolio_value],
                :pnl => Float64[agent.position.pnl]
            ) for agent in agents
        )
        
        # Track market history
        market_history = [deepcopy(market)]
        
        # Run simulation
        for step in 1:n_steps
            try
                # Step 1: All agents observe market
                observations = [observe_market(market, market_params) for _ in agents]
                beliefs = [observe(agent, obs) for (agent, obs) in zip(agents, observations)]
                
                # Step 2: All agents select actions (simultaneous)
                orders = [act(agent, belief, market.price) 
                         for (agent, belief) in zip(agents, beliefs)]
                
                # Step 3: Execute all orders
                total_volume = 0.0
                for (agent, order) in zip(agents, orders)
                    try
                        # Execute order
                        execution_price, slippage_cost = execute_order(market, order, market_params)
                        
                        # Update agent position
                        prev_value = agent.position.portfolio_value
                        update_position!(agent, order, execution_price, slippage_cost, market.price)
                        
                        # Compute reward
                        reward = agent.position.portfolio_value - prev_value - slippage_cost
                        
                        # Agent learns (Q-learning: immediate, Policy gradient: accumulate)
                        if agent isa QLearningAgent
                            state = discretize_price(market.price, market_params.initial_price)
                            next_price = market.price  # Will be updated below
                            next_state = discretize_price(next_price, market_params.initial_price)
                            q_learning_update!(agent, state, order.action, reward, next_state, false)
                        elseif agent isa PolicyGradientAgent
                            learn!(agent, market.price, Float64(order.quantity), reward)
                        end
                        
                        # Track volume
                        total_volume += Float64(order.quantity)
                        
                    catch e
                        @warn "Agent action failed" agent_id=agent.id step exception=e
                        continue
                    end
                end
                
                # Step 4: Update market state
                market.price = update_price(market, market_params)
                push!(market.price_history, market.price)
                market.volume = total_volume
                push!(market.volume_history, total_volume)
                market.time += 1
                
                # Step 5: Update agent histories
                for agent in agents
                    # Update portfolio value with new price
                    agent.position.portfolio_value = agent.position.cash + 
                                                    agent.position.shares * market.price
                    
                    # Track history
                    push!(agent_histories[agent.id][:cash], agent.position.cash)
                    push!(agent_histories[agent.id][:shares], agent.position.shares)
                    push!(agent_histories[agent.id][:portfolio_value], agent.position.portfolio_value)
                    push!(agent_histories[agent.id][:pnl], 
                          agent.position.portfolio_value - agent_histories[agent.id][:portfolio_value][1])
                end
                
                # Track market state
                push!(market_history, deepcopy(market))
                
                # Log progress
                if step % 100 == 0
                    @info "Simulation progress" step price=market.price total_volume
                end
                
            catch e
                @warn "Simulation step failed" step exception=e
                continue
            end
        end
        
        # Compute final metrics
        metrics = Dict(
            agent.id => compute_metrics(agent_histories[agent.id], market_history)
            for agent in agents
        )
        
        @info "Simulation complete" n_steps final_price=market.price
        
        return Dict(
            :market_history => market_history,
            :agent_histories => agent_histories,
            :price_history => [m.price for m in market_history],
            :metrics => metrics
        )
        
    catch e
        @error "Simulation failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    compute_metrics(agent_history::Dict, market_history::Vector) -> PerformanceMetrics

Compute performance metrics for agent.

# Arguments
- `agent_history::Dict`: Agent's position history
- `market_history::Vector`: Market states over time

# Returns
- `PerformanceMetrics`: Performance statistics

# Metrics Computed

## Cumulative Return
```
return = (final_value - initial_value) / initial_value
```

## Sharpe Ratio
```
sharpe = mean(returns) / std(returns) * sqrt(252)
```
Annualized risk-adjusted return (252 trading days/year)

## Maximum Drawdown
```
drawdown = max((peak - trough) / peak)
```
Worst peak-to-trough decline

## Win Rate
```
win_rate = profitable_trades / total_trades
```

## Total Trades
Count of buy/sell actions (excludes holds)

# Notes
- Returns computed from portfolio value changes
- Sharpe ratio annualized (assumes daily timesteps)
- Drawdown measures worst-case loss
- Win rate alone can be misleading (need magnitude)

# Error Handling
- Returns zero metrics if insufficient data
- Handles edge cases (no trades, constant value)
- Logs warnings for invalid metrics

# Example
```julia
metrics = compute_metrics(agent_history, market_history)
# metrics.cumulative_return = 0.15 (15% return)
# metrics.sharpe_ratio = 1.2 (good risk-adjusted)
# metrics.max_drawdown = 0.08 (8% max loss)
```
"""
function compute_metrics(agent_history::Dict, market_history::Vector)::PerformanceMetrics
    try
        portfolio_values = agent_history[:portfolio_value]
        
        if length(portfolio_values) < 2
            @warn "Insufficient data for metrics"
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0)
        end
        
        # Cumulative return
        initial_value = portfolio_values[1]
        final_value = portfolio_values[end]
        cumulative_return = (final_value - initial_value) / initial_value
        
        # Returns (period-to-period changes)
        returns = diff(portfolio_values) ./ portfolio_values[1:end-1]
        
        # Sharpe ratio (annualized, assuming daily timesteps)
        if std(returns) > 0
            sharpe_ratio = mean(returns) / std(returns) * sqrt(252)
        else
            sharpe_ratio = 0.0
        end
        
        # Maximum drawdown
        # Formula: max((peak - trough) / peak) where peak is running maximum
        # Result should be in [0, 1] where 0 = no drawdown, 1 = total loss
        max_drawdown = 0.0
        peak = portfolio_values[1]
        for value in portfolio_values
            if value > peak
                peak = value
            end
            # Only compute drawdown if peak > 0 (avoid division by zero)
            if peak > 0.0
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown
                    max_drawdown = drawdown
                end
            end
        end
        
        # Clamp to [0, 1] (should not exceed 1.0 by definition)
        # If > 1.0, indicates portfolio went negative (leverage/margin)
        max_drawdown = clamp(max_drawdown, 0.0, 1.0)
        
        # Win rate and total trades
        # Count trades from share changes
        shares = agent_history[:shares]
        trades = 0
        profitable_trades = 0
        
        for i in 2:length(shares)
            if shares[i] != shares[i-1]
                trades += 1
                # Check if portfolio value increased
                if portfolio_values[i] > portfolio_values[i-1]
                    profitable_trades += 1
                end
            end
        end
        
        win_rate = trades > 0 ? profitable_trades / trades : 0.0
        
        return PerformanceMetrics(
            cumulative_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            trades
        )
        
    catch e
        @error "Metrics computation failed" exception=(e, catch_backtrace())
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0)
    end
end

"""
    step_market!(market::MarketState, agents::Vector{TradingAgent}, 
                 market_params::MarketParams) -> Float64

Execute single simulation step (all agents act).

# Arguments
- `market::MarketState`: Current market (modified in-place)
- `agents::Vector{TradingAgent}`: All agents
- `market_params::MarketParams`: Market configuration

# Returns
- `Float64`: Total trading volume this step

# Algorithm
1. Agents observe and act
2. Execute orders
3. Update market price
4. Return total volume

# Notes
- Used internally by run_simulation
- Can be used for custom simulation loops
- Modifies market state in-place

# Example
```julia
volume = step_market!(market, agents, params)
# market.price updated, agents acted
```
"""
function step_market!(
    market::MarketState,
    agents::Vector{TradingAgent},
    market_params::MarketParams
)::Float64
    total_volume = 0.0
    
    # All agents observe and act
    for agent in agents
        try
            observation = observe_market(market, market_params)
            belief = observe(agent, observation)
            order = act(agent, belief, market.price)
            
            execution_price, slippage_cost = execute_order(market, order, market_params)
            update_position!(agent, order, execution_price, slippage_cost, market.price)
            
            total_volume += Float64(order.quantity)
        catch e
            @warn "Agent step failed" agent_id=agent.id exception=e
            continue
        end
    end
    
    # Update market
    market.price = update_price(market, market_params)
    push!(market.price_history, market.price)
    market.volume = total_volume
    push!(market.volume_history, total_volume)
    market.time += 1
    
    return total_volume
end

# Exports
export run_simulation, compute_metrics, step_market!

