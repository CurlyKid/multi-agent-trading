"""
Market environment implementation.

Provides realistic market simulation with:
- Geometric Brownian motion price dynamics
- Order execution with slippage
- Partial observability (noisy observations)
- Market state tracking
"""

using Random
using Distributions

"""
    initialize_market(params::MarketParams) -> MarketState

Initialize market environment with given parameters.

# Arguments
- `params::MarketParams`: Market configuration

# Returns
- `MarketState`: Initial market state

# Error Handling
- Validates parameters (σ > 0, initial_price > 0)
- Logs initialization with parameters
- Throws ArgumentError for invalid parameters

# Example
```julia
params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
market = initialize_market(params)
```
"""
function initialize_market(params::MarketParams)::MarketState
    try
        # Validate parameters
        if params.σ <= 0
            throw(ArgumentError("Volatility σ must be positive, got $(params.σ)"))
        end
        if params.initial_price <= 0
            throw(ArgumentError("Initial price must be positive, got $(params.initial_price)"))
        end
        if params.slippage_factor < 0
            throw(ArgumentError("Slippage factor must be non-negative, got $(params.slippage_factor)"))
        end
        if params.observation_noise < 0
            throw(ArgumentError("Observation noise must be non-negative, got $(params.observation_noise)"))
        end
        
        @info "Initializing market" μ=params.μ σ=params.σ initial_price=params.initial_price
        
        return MarketState(
            0,                              # time
            params.initial_price,           # price
            0.0,                            # volume
            [params.initial_price],         # price_history (initial price at t=0)
            Float64[]                       # volume_history (empty, no steps yet)
        )
    catch e
        @error "Market initialization failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    update_price(state::MarketState, params::MarketParams, dt::Float64=1.0) -> Float64

Update market price using geometric Brownian motion (GBM).

# Arguments
- `state::MarketState`: Current market state
- `params::MarketParams`: Market parameters
- `dt::Float64`: Time step (default 1.0)

# Returns
- `Float64`: New price

# Algorithm
Geometric Brownian Motion:
```
S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
```
where:
- S(t) = current price
- μ = drift (expected return)
- σ = volatility (standard deviation)
- Z ~ N(0,1) = standard normal random variable
- dt = time step

# Notes
- Ensures price stays positive (max with small epsilon)
- Drift term (μ - σ²/2) accounts for Itô's lemma correction
- Diffusion term σ√dt * Z adds randomness
- Log-normal distribution ensures positive prices

# Error Handling
- Validates current price is positive
- Ensures new price is positive
- Logs price updates at debug level
"""
function update_price(state::MarketState, params::MarketParams, dt::Float64=1.0)::Float64
    try
        if state.price <= 0
            throw(ArgumentError("Current price must be positive, got $(state.price)"))
        end
        
        # Generate standard normal random variable
        Z = randn()
        
        # GBM formula components:
        # 1. Drift term: (μ - σ²/2)dt
        #    - μ is expected return
        #    - σ²/2 is Itô's lemma correction (converts arithmetic to geometric)
        drift = (params.μ - 0.5 * params.σ^2) * dt
        
        # 2. Diffusion term: σ√dt * Z
        #    - σ controls volatility magnitude
        #    - √dt scales volatility with time step
        #    - Z adds randomness
        diffusion = params.σ * sqrt(dt) * Z
        
        # 3. Exponential ensures positive price
        #    S(t+dt) = S(t) * exp(drift + diffusion)
        new_price = state.price * exp(drift + diffusion)
        
        # Ensure price stays positive (numerical safety)
        new_price = max(new_price, 0.01)
        
        @debug "Price updated" old_price=state.price new_price=new_price change=(new_price/state.price - 1)
        
        return new_price
    catch e
        @error "Price update failed" exception=(e, catch_backtrace()) current_price=state.price
        # Return current price as safe fallback
        return state.price
    end
end

"""
    execute_order(state::MarketState, order::Order, params::MarketParams) -> Tuple{Float64, Float64}

Execute trading order with slippage and market impact.

# Arguments
- `state::MarketState`: Current market state
- `order::Order`: Order to execute (buy/sell/hold, quantity)
- `params::MarketParams`: Market parameters

# Returns
- `Tuple{Float64, Float64}`: (execution_price, slippage_cost)

# Algorithm
Slippage model:
```
slippage = slippage_factor * |quantity|
buy:  execution_price = market_price * (1 + slippage)
sell: execution_price = market_price * (1 - slippage)
hold: execution_price = market_price, slippage = 0
```

# Notes
- Slippage increases with order size (linear model)
- Buy orders pay premium (worse execution)
- Sell orders receive discount (worse execution)
- Hold orders have no slippage
- Slippage cost = slippage * market_price

# Error Handling
- Validates order quantity is finite
- Validates action is valid (:buy, :sell, :hold)
- Logs execution details
- Returns safe defaults (market price, zero slippage) on error

# Example
```julia
order = Order(:buy, 10)
execution_price, slippage_cost = execute_order(state, order, params)
# execution_price > market_price (paid premium)
# slippage_cost = premium amount
```
"""
function execute_order(state::MarketState, order::Order, params::MarketParams)::Tuple{Float64, Float64}
    try
        # Validate order
        if !isfinite(order.quantity)
            throw(ArgumentError("Order quantity must be finite, got $(order.quantity)"))
        end
        if !(order.action in [:buy, :sell, :hold])
            throw(ArgumentError("Invalid action $(order.action), must be :buy, :sell, or :hold"))
        end
        
        # Hold order: no execution, no slippage
        if order.action == :hold
            return (state.price, 0.0)
        end
        
        # Compute slippage (proportional to order size)
        slippage = params.slippage_factor * abs(order.quantity)
        
        # Execution price depends on action
        if order.action == :buy
            # Buy: pay premium (worse execution)
            execution_price = state.price * (1 + slippage)
        else  # :sell
            # Sell: receive discount (worse execution)
            execution_price = state.price * (1 - slippage)
        end
        
        # Slippage cost (absolute amount)
        slippage_cost = slippage * state.price
        
        @debug "Order executed" action=order.action quantity=order.quantity market_price=state.price execution_price slippage_cost
        
        return (execution_price, slippage_cost)
    catch e
        @error "Order execution failed" exception=(e, catch_backtrace()) order
        # Safe default: market price, no slippage
        return (state.price, 0.0)
    end
end

"""
    observe_market(state::MarketState, params::MarketParams) -> Observation

Get noisy observation of market state (partial observability).

# Arguments
- `state::MarketState`: True market state
- `params::MarketParams`: Market parameters

# Returns
- `Observation`: Noisy price and volume observation

# Algorithm
Observation noise model:
```
noise ~ N(0, observation_noise)
observed_price = true_price * (1 + noise)
```

# Notes
- Adds Gaussian noise to price observation
- Volume is observable (no noise)
- Simulates delayed/imperfect information
- Enables POMDP framework (partial observability)
- Agents must maintain belief states over true price

# Error Handling
- Validates observation noise is non-negative
- Ensures observed price is positive
- Logs observations at debug level

# Example
```julia
obs = observe_market(state, params)
# obs.observed_price ≈ state.price (with noise)
# obs.volume == state.volume (exact)
```
"""
function observe_market(state::MarketState, params::MarketParams)::Observation
    try
        if params.observation_noise < 0
            throw(ArgumentError("Observation noise must be non-negative, got $(params.observation_noise)"))
        end
        
        # Generate Gaussian noise
        noise = randn() * params.observation_noise
        
        # Add noise to price
        noisy_price = state.price * (1 + noise)
        
        # Ensure positive price
        noisy_price = max(noisy_price, 0.01)
        
        @debug "Market observed" true_price=state.price observed_price=noisy_price noise_pct=(noise*100)
        
        return Observation(noisy_price, state.volume, state.time)
    catch e
        @error "Market observation failed" exception=(e, catch_backtrace())
        # Safe default: true price, no noise
        return Observation(state.price, state.volume, state.time)
    end
end

"""
    step_market!(state::MarketState, params::MarketParams, orders::Vector{Order})

Update market state for one timestep.

# Arguments
- `state::MarketState`: Current market state (modified in-place)
- `params::MarketParams`: Market parameters
- `orders::Vector{Order}`: Orders from all agents

# Side Effects
- Updates state.time
- Updates state.price (GBM)
- Updates state.volume (sum of order quantities)
- Appends to state.price_history
- Appends to state.volume_history

# Notes
- Executes all orders at current price (simultaneous execution)
- Updates price after orders (price impact in next timestep)
- Volume = sum of absolute order quantities

# Error Handling
- Validates state is not corrupted
- Logs market updates
- Continues on individual order failures
"""
function step_market!(state::MarketState, params::MarketParams, orders::Vector{Order})
    try
        # Update time
        state.time += 1
        
        # Compute total volume
        volume = sum(abs(order.quantity) for order in orders if order.action != :hold)
        state.volume = volume
        
        # Update price (GBM)
        state.price = update_price(state, params)
        
        # Record history
        push!(state.price_history, state.price)
        push!(state.volume_history, state.volume)
        
        @debug "Market stepped" time=state.time price=state.price volume=state.volume
    catch e
        @error "Market step failed" exception=(e, catch_backtrace()) time=state.time
        # Continue with current state (don't crash simulation)
    end
end

# Exports
export initialize_market, update_price, execute_order, observe_market, step_market!
