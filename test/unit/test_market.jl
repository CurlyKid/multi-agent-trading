"""
Unit Tests: Market Environment

Tests specific examples and edge cases for market dynamics:
- Price positivity (GBM never produces negative prices)
- GBM statistics (drift and volatility match parameters)
- Slippage computation (price impact proportional to order size)
- Observation noise (noisy observations within bounds)

Run with: julia --project=. test/unit/test_market.jl
"""

using Test
using MultiAgentTrading
using Statistics
using Random

@testset "Market Environment Tests" begin
    
    # ========================================================================
    # Test 1: Price Positivity
    # ========================================================================
    
    @testset "Price Positivity" begin
        # GBM should never produce negative prices
        params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        # Run 1000 price updates
        for _ in 1:1000
            new_price = update_price(state, params)
            @test new_price > 0.0  # Price must be positive
            state.price = new_price
        end
        
        # Test with extreme volatility
        params_extreme = MarketParams(0.0, 0.5, 100.0, 0.01, 0.005)
        state_extreme = initialize_market(params_extreme)
        
        for _ in 1:100
            new_price = update_price(state_extreme, params_extreme)
            @test new_price > 0.0  # Even with high volatility
            state_extreme.price = new_price
        end
    end
    
    # ========================================================================
    # Test 2: GBM Statistics
    # ========================================================================
    
    @testset "GBM Statistics" begin
        # Test that drift and volatility match parameters (approximately)
        Random.seed!(42)  # Reproducibility
        
        params = MarketParams(0.001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        n_steps = 10000
        prices = Float64[state.price]
        
        for _ in 1:n_steps
            new_price = update_price(state, params)
            push!(prices, new_price)
            state.price = new_price
        end
        
        # Compute log returns
        log_returns = diff(log.(prices))
        
        # Expected mean: μ - σ²/2 (Itô's lemma correction)
        expected_mean = params.μ - params.σ^2 / 2
        observed_mean = mean(log_returns)
        
        # Expected std: σ
        expected_std = params.σ
        observed_std = std(log_returns)
        
        # Allow 30% tolerance (statistical variation with 10k samples)
        @test isapprox(observed_mean, expected_mean, rtol=0.3)
        @test isapprox(observed_std, expected_std, rtol=0.3)
    end
    
    # ========================================================================
    # Test 3: Slippage Computation
    # ========================================================================
    
    @testset "Slippage Computation" begin
        params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        # Test buy order (positive quantity)
        order_buy = Order(:buy, 10)  # Buy 10 shares
        execution_price_buy, slippage_buy = execute_order(state, order_buy, params)
        
        # Buy order should increase price
        @test execution_price_buy > state.price
        @test slippage_buy > 0.0
        
        # Slippage cost = slippage_factor * quantity * price
        expected_slippage_buy = params.slippage_factor * abs(order_buy.quantity) * state.price
        @test isapprox(slippage_buy, expected_slippage_buy, rtol=1e-6)
        
        # Test sell order (negative quantity)
        order_sell = Order(:sell, 10)  # Sell 10 shares
        execution_price_sell, slippage_sell = execute_order(state, order_sell, params)
        
        # Sell order should decrease price
        @test execution_price_sell < state.price
        @test slippage_sell > 0.0
        
        # Slippage cost = slippage_factor * quantity * price
        expected_slippage_sell = params.slippage_factor * abs(order_sell.quantity) * state.price
        @test isapprox(slippage_sell, expected_slippage_sell, rtol=1e-6)
        
        # Test zero order (no slippage)
        order_zero = Order(:hold, 0)
        execution_price_zero, slippage_zero = execute_order(state, order_zero, params)
        
        @test execution_price_zero == state.price
        @test slippage_zero == 0.0
        
        # Test large order (higher slippage)
        order_large = Order(:buy, 100)
        _, slippage_large = execute_order(state, order_large, params)
        
        order_small = Order(:buy, 10)
        _, slippage_small = execute_order(state, order_small, params)
        
        @test slippage_large > slippage_small
    end
    
    # ========================================================================
    # Test 4: Observation Noise
    # ========================================================================
    
    @testset "Observation Noise" begin
        params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        # Generate multiple observations
        n_obs = 1000
        observations = [observe_market(state, params) for _ in 1:n_obs]
        
        # Extract noisy prices
        noisy_prices = [obs.observed_price for obs in observations]
        
        # Mean should be close to true price
        @test isapprox(mean(noisy_prices), state.price, rtol=0.01)
        
        # Std should be close to observation noise
        observed_noise = std(noisy_prices)
        expected_noise = params.observation_noise * state.price
        @test isapprox(observed_noise, expected_noise, rtol=0.2)
        
        # All observations should be positive
        @test all(p -> p > 0.0, noisy_prices)
        
        # Test with zero noise
        params_no_noise = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.0)
        state_no_noise = initialize_market(params_no_noise)
        
        obs_no_noise = observe_market(state_no_noise, params_no_noise)
        @test obs_no_noise.observed_price == state_no_noise.price  # Exact match
    end
    
    # ========================================================================
    # Test 5: Market Initialization
    # ========================================================================
    
    @testset "Market Initialization" begin
        params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        # Check initial values
        @test state.price == params.initial_price
        @test state.time == 0
        @test state.volume == 0.0
        
        # Test with different initial price
        params2 = MarketParams(0.0001, 0.02, 50.0, 0.01, 0.005)
        state2 = initialize_market(params2)
        @test state2.price == 50.0
    end
    
    # ========================================================================
    # Test 6: Step Market
    # ========================================================================
    
    @testset "Step Market" begin
        params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
        state = initialize_market(params)
        
        initial_price = state.price
        initial_time = state.time
        
        # Step market forward (with empty orders)
        step_market!(state, params, Order[])
        
        # Time should increment
        @test state.time == initial_time + 1
        
        # Price should change (with high probability)
        # Note: Could be same with very low probability
        @test state.price > 0.0
        
        # Volume should remain 0 (no orders)
        @test state.volume == 0.0
    end
    
end

println("\n✅ All market unit tests passed!")
