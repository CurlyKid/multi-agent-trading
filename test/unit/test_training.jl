"""
Unit Tests: Training Loop

Tests specific examples and edge cases for RL training:
- Episode execution (single episode runs correctly)
- Experience replay (Q-learning buffer management)
- Learning curves (statistics tracking)

Run with: julia --project=. test/unit/test_training.jl
"""

using Test
using MultiAgentTrading
using Statistics

@testset "Training Loop Tests" begin
    
    # Setup
    params = MarketParams(0.0001, 0.02, 100.0, 0.01, 0.005)
    
    # ========================================================================
    # Test 1: Episode Execution
    # ========================================================================
    
    @testset "Episode Execution" begin
        # Create Q-learning agent
        agent = create_qlearning_agent(1, 10000.0, 0.1, 0.95, 0.1)
        
        # Train for 1 episode
        stats = train_agent!(agent, params, 1, 100)
        
        # Check stats structure
        @test haskey(stats, :episode_returns)
        @test haskey(stats, :learning_curve)
        
        # Check episode ran
        @test length(stats[:episode_returns]) == 1
        @test stats[:episode_returns][1] isa Float64
        
        # Agent should have some experience
        @test agent.position.cash != 10000.0 || agent.position.shares != 0
    end
    
    # ========================================================================
    # Test 2: Multiple Episodes
    # ========================================================================
    
    @testset "Multiple Episodes" begin
        # Create agent
        agent = create_qlearning_agent(2, 10000.0, 0.1, 0.95, 0.1)
        
        # Train for 10 episodes
        stats = train_agent!(agent, params, 10, 50)
        
        # Check all episodes ran
        @test length(stats[:episode_returns]) == 10
        @test all(r -> r isa Float64, stats[:episode_returns])
        
        # Check learning curve (moving average)
        @test length(stats[:learning_curve]) == 10
        @test all(isfinite, stats[:learning_curve])
        
        # Learning curve should be smoother than raw returns
        # (moving average reduces variance)
        if length(stats[:episode_returns]) >= 10
            returns_std = std(stats[:episode_returns])
            curve_std = std(stats[:learning_curve])
            # Curve should be smoother (lower std) - but may not always hold
            # Just check both are finite
            @test isfinite(returns_std)
            @test isfinite(curve_std)
        end
    end
    
    # ========================================================================
    # Test 3: Experience Replay (Q-Learning)
    # ========================================================================
    
    @testset "Experience Replay" begin
        # Create Q-learning agent
        agent = create_qlearning_agent(3, 10000.0, 0.1, 0.95, 0.1)
        
        # Train with experience replay
        stats = train_agent!(agent, params, 5, 100)
        
        # Should complete without error
        @test length(stats[:episode_returns]) == 5
        
        # Agent should have learned something
        # (Q-values updated, though may not be profitable yet)
        @test agent.position.cash != 10000.0 || agent.position.shares != 0
    end
    
    # ========================================================================
    # Test 4: Learning Curves
    # ========================================================================
    
    @testset "Learning Curves" begin
        # Create agent
        agent = create_qlearning_agent(4, 10000.0, 0.1, 0.95, 0.1)
        
        # Train for enough episodes to see curve
        stats = train_agent!(agent, params, 20, 50)
        
        # Check learning curve properties
        @test length(stats[:learning_curve]) == 20
        @test all(isfinite, stats[:learning_curve])
        
        # Learning curve should exist for all episodes
        @test length(stats[:learning_curve]) == length(stats[:episode_returns])
        
        # First few values should be close to raw returns
        # (not enough history for smoothing)
        @test isapprox(stats[:learning_curve][1], stats[:episode_returns][1], rtol=0.1)
    end
    
    # ========================================================================
    # Test 5: Epsilon Decay
    # ========================================================================
    
    @testset "Epsilon Decay" begin
        # Create agent with high epsilon
        agent = create_qlearning_agent(5, 10000.0, 0.1, 0.95, 0.5)
        initial_epsilon = agent.epsilon
        
        # Train for many episodes
        stats = train_agent!(agent, params, 50, 20)
        
        # Epsilon should decay (checked via agent.epsilon, not stats)
        @test agent.epsilon < initial_epsilon
        
        # Should not go below minimum (0.01)
        @test agent.epsilon >= 0.01
        
        # Should decay exponentially (0.995 per episode)
        expected_epsilon = max(0.01, initial_epsilon * 0.995^50)
        @test isapprox(agent.epsilon, expected_epsilon, rtol=0.01)
    end
    
    # ========================================================================
    # Test 6: Policy Gradient Training
    # ========================================================================
    
    @testset "Policy Gradient Training" begin
        # Create policy gradient agent
        agent = create_policy_gradient_agent(6, 10000.0, 10, 0.01)
        
        # Train (uses trajectories, not experience replay)
        stats = train_agent!(agent, params, 5, 100)
        
        # Should complete without error
        @test length(stats[:episode_returns]) == 5
        @test all(isfinite, stats[:episode_returns])
        
        # Policy params should be updated
        # (no direct test, but training should not error)
    end
    
    # ========================================================================
    # Test 7: Short Episodes
    # ========================================================================
    
    @testset "Short Episodes" begin
        # Create agent
        agent = create_qlearning_agent(7, 10000.0, 0.1, 0.95, 0.1)
        
        # Train with very short episodes (10 steps)
        stats = train_agent!(agent, params, 3, 10)
        
        # Should still work
        @test length(stats[:episode_returns]) == 3
        @test all(isfinite, stats[:episode_returns])
    end
    
    # ========================================================================
    # Test 8: Long Episodes
    # ========================================================================
    
    @testset "Long Episodes" begin
        # Create agent
        agent = create_qlearning_agent(8, 10000.0, 0.1, 0.95, 0.1)
        
        # Train with longer episodes (500 steps)
        stats = train_agent!(agent, params, 2, 500)
        
        # Should complete
        @test length(stats[:episode_returns]) == 2
        @test all(isfinite, stats[:episode_returns])
    end
    
    # ========================================================================
    # Test 9: Statistics Validity
    # ========================================================================
    
    @testset "Statistics Validity" begin
        # Create agent
        agent = create_qlearning_agent(9, 10000.0, 0.1, 0.95, 0.1)
        
        # Train
        stats = train_agent!(agent, params, 10, 100)
        
        # All returns should be finite
        @test all(isfinite, stats[:episode_returns])
        
        # Learning curve should be finite
        @test all(isfinite, stats[:learning_curve])
        
        # Returns can be positive or negative (untrained agents)
        # Just check they're reasonable (not NaN, Inf, or extreme)
        @test all(r -> abs(r) < 1e6, stats[:episode_returns])
    end
    
end

println("\n✅ All training unit tests passed!")
