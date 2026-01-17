"""
Test Runner: Multi-Agent RL Trading Simulator

Runs all test suites:
- Unit tests (market, POMDP, agents, training)
- Integration tests (full simulation)
- Property tests (market, POMDP invariants)

Run with: julia --project=. test/runtests.jl
Or: julia --project=. -e "using Pkg; Pkg.test()"
"""

using Test
using MultiAgentTrading

@testset "Multi-Agent Trading Test Suite" begin
    
    # ========================================================================
    # Unit Tests
    # ========================================================================
    
    @testset "Unit Tests" begin
        @testset "Market Environment" begin
            include("unit/test_market.jl")
        end
        
        @testset "POMDP Framework" begin
            include("unit/test_pomdp.jl")
        end
        
        @testset "Trading Agents" begin
            include("unit/test_agents.jl")
        end
        
        @testset "Training Loop" begin
            include("unit/test_training.jl")
        end
    end
    
    # ========================================================================
    # Integration Tests
    # ========================================================================
    
    @testset "Integration Tests" begin
        include("integration/test_full_simulation.jl")
    end
    
    # ========================================================================
    # Property Tests
    # ========================================================================
    
    @testset "Property Tests" begin
        @testset "Market Properties" begin
            include("property/test_market_properties.jl")
        end
        
        @testset "POMDP Properties" begin
            include("property/test_pomdp_properties.jl")
        end
    end
    
end

println("\n" * "="^70)
println("✅ ALL TESTS PASSED!")
println("="^70)
println("\nTest Coverage:")
println("  - Unit Tests: Market, POMDP, Agents, Training")
println("  - Integration Tests: Full simulation pipeline")
println("  - Property Tests: Universal invariants (20,000+ iterations)")
println("\nFramework validated. Ready for deployment.")
println("="^70)
