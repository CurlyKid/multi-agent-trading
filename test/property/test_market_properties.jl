"""
Property Tests: Market Dynamics Invariants

Tests universal properties that should hold for ALL inputs:
- Property 1: Market Dynamics Invariants (Requirements 1.1-1.5)

Property tests run 100 iterations with random inputs to validate
correctness across the entire input space.

Run with: julia --project=. test/property/test_market_properties.jl
"""

using Test
using MultiAgentTrading
using Statistics
using Random

@testset "Market Dynamics Property Tests" begin
    
    # ========================================================================
    # Property 1: Market Dynamics Invariants
    # ========================================================================
    
    @testset "Property 1: Market Dynamics Invariants" begin
        # Test with 100 random parameter combinations
        Random.seed!(42)
        
        for trial in 1:100
            # Generate random market parameters
            μ = rand() * 0.001  # Drift: [0, 0.001]
            σ = rand() * 0.1    # Volatility: [0, 0.1]
            initial_price = 50.0 + rand() * 150.0  # Price: [50, 200]
            slippage_factor = rand() * 0.02  # Slippage: [0, 0.02]
            observation_noise = rand() * 0.01  # Noise: [0, 0.01]
            
            params = MarketParams(μ, σ, initial_price, slippage_factor, observation_noise)
            state = initialize_market(params)
            
            # Property 1.1: Price positivity (GBM never produces negative prices)
            for step in 1:100
                new_price = update_price(state, params)
                @test new_price > 0.0
                state.price = new_price
            end
            
            # Property 1.2: Slippage proportional to order size
            state = initialize_market(params)  # Reset
            
            order_small = Order(:buy, 1)
            order_large = Order(:buy, 100)
            
            _, slippage_small = execute_order(state, order_small, params)
            _, slippage_large = execute_order(state, order_large, params)
            
            @test slippage_large > slippage_small
            @test slippage_small ≈ slippage_factor * 1 * state.price rtol=1e-6
            @test slippage_large ≈ slippage_factor * 100 * state.price rtol=1e-6
            
            # Property 1.3: Observation noise bounded
            observations = [observe_market(state, params) for _ in 1:50]
            noisy_prices = [obs.observed_price for obs in observations]
            
            # All observations should be positive
            @test all(p -> p > 0.0, noisy_prices)
            
            # Mean should be close to true price (within 3σ)
            if observation_noise > 0.0
                expected_std = observation_noise * state.price
                @test abs(mean(noisy_prices) - state.price) < 3 * expected_std
            else
                # Zero noise → exact observations
                @test all(p -> p == state.price, noisy_prices)
            end
            
            # Property 1.4: Market state consistency
            @test state.time >= 0
            @test state.volume >= 0.0
            @test length(state.price_history) == state.time + 1
            @test length(state.volume_history) == state.time
            
            # Property 1.5: Order execution symmetry
            state = initialize_market(params)  # Reset
            
            order_buy = Order(:buy, 10)
            order_sell = Order(:sell, 10)
            
            price_buy, _ = execute_order(state, order_buy, params)
            price_sell, _ = execute_order(state, order_sell, params)
            
            # Buy increases price, sell decreases price
            @test price_buy > state.price
            @test price_sell < state.price
            
            # Magnitude should be symmetric
            @test abs(price_buy - state.price) ≈ abs(state.price - price_sell) rtol=0.01
        end
    end
    
    # ========================================================================
    # Property 2: GBM Statistical Properties
    # ========================================================================
    
    @testset "Property 2: GBM Statistical Properties" begin
        # Note: Statistical tests may occasionally fail due to random sampling
        # This is expected behavior - GBM is stochastic
        Random.seed!(42)
        
        for trial in 1:20  # Fewer trials (statistical tests are expensive)
            μ = rand() * 0.001
            σ = rand() * 0.05
            initial_price = 100.0
            
            params = MarketParams(μ, σ, initial_price, 0.01, 0.005)
            state = initialize_market(params)
            
            # Generate long price trajectory
            n_steps = 5000
            prices = Float64[state.price]
            
            for _ in 1:n_steps
                new_price = update_price(state, params)
                push!(prices, new_price)
                state.price = new_price
            end
            
            # Compute log returns
            log_returns = diff(log.(prices))
            
            # Expected mean: μ - σ²/2 (Itô's lemma correction)
            expected_mean = μ - σ^2 / 2
            observed_mean = mean(log_returns)
            
            # Expected std: σ
            expected_std = σ
            observed_std = std(log_returns)
            
            # Allow 60% tolerance (statistical variation with finite samples)
            @test isapprox(observed_mean, expected_mean, rtol=0.6)
            @test isapprox(observed_std, expected_std, rtol=0.6)
        end
    end
    
    # ========================================================================
    # Property 3: Market Impact Consistency
    # ========================================================================
    
    @testset "Property 3: Market Impact Consistency" begin
        Random.seed!(42)
        
        for trial in 1:100
            params = MarketParams(0.0001, 0.02, 100.0, rand() * 0.02, 0.005)
            state = initialize_market(params)
            
            # Property: Larger orders → larger price impact
            quantities = [1, 5, 10, 50, 100]
            impacts = Float64[]
            
            for qty in quantities
                order = Order(:buy, qty)
                execution_price, _ = execute_order(state, order, params)
                impact = abs(execution_price - state.price)
                push!(impacts, impact)
            end
            
            # Impacts should be monotonically increasing
            for i in 2:length(impacts)
                @test impacts[i] >= impacts[i-1]
            end
            
            # Impact should be proportional to quantity
            @test impacts[end] / impacts[1] ≈ quantities[end] / quantities[1] rtol=0.01
        end
    end
    
end

println("\n✅ All market property tests passed!")
