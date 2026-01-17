"""
POMDP framework implementation.

Provides belief state management for trading under uncertainty:
- Belief state representation (probability distributions)
- Bayes' rule updates
- Observation likelihood computation
- Belief normalization and validation
"""

using Distributions
using LinearAlgebra

"""
    update_belief(prior::BeliefState, observation::Observation, 
                  transition_model::Function) -> BeliefState

Update belief state using Bayes' rule.

# Arguments
- `prior::BeliefState`: Prior belief distribution
- `observation::Observation`: New observation
- `transition_model::Function`: State transition model (unused in current implementation)

# Returns
- `BeliefState`: Posterior belief distribution

# Algorithm
Bayes' rule for belief update:
```
P(state | observation) ∝ P(observation | state) * P(state)
```

Step-by-step:
1. Compute likelihood P(obs|state) for each possible state
2. Multiply by prior P(state) to get unnormalized posterior
3. Normalize to ensure probabilities sum to 1

Mathematical details:
- Likelihood: How probable is this observation given each state?
- Prior: What did we believe before seeing the observation?
- Posterior: What should we believe after seeing the observation?

# Notes
- Handles numerical stability (checks for zero total)
- Falls back to uniform belief if update fails
- Validates prior is valid probability distribution
- Ensures posterior is valid probability distribution

# Error Handling
- Validates prior sums to 1 (within tolerance)
- Ensures posterior is valid (non-negative, sums to 1)
- Logs belief updates at debug level
- Returns uniform belief as safe fallback on error

# Example
```julia
prior = BeliefState(states, [0.3, 0.4, 0.3])
obs = Observation(102.5, 100.0, 1)
posterior = update_belief(prior, obs, transition_model)
# posterior.weights updated based on observation likelihood
```
"""
function update_belief(
    prior::BeliefState,
    observation::Observation,
    transition_model::Function=identity
)::BeliefState
    try
        # Validate prior is valid belief
        prior_sum = sum(prior.weights)
        if !isapprox(prior_sum, 1.0, atol=1e-6)
            @warn "Prior belief does not sum to 1" prior_sum
            # Normalize prior
            prior_weights = prior.weights ./ prior_sum
        else
            prior_weights = prior.weights
        end
        
        # Step 1: Compute likelihoods P(observation | state) for each state
        # This measures how probable the observation is under each hypothetical state
        likelihoods = [observation_likelihood(observation, state) for state in prior.states]
        
        # Step 2: Apply Bayes' rule (unnormalized)
        # posterior ∝ likelihood * prior
        # This combines:
        # - What we observed (likelihood)
        # - What we believed before (prior)
        posterior_weights = likelihoods .* prior_weights
        
        # Step 3: Normalize to get valid probability distribution
        # Ensures sum(posterior_weights) = 1.0
        total = sum(posterior_weights)
        
        if total > 0
            # Normal case: normalize
            posterior_weights ./= total
        else
            # Edge case: all likelihoods zero (observation impossible under all states)
            # Fall back to uniform distribution
            @warn "All observation likelihoods zero, using uniform belief"
            n = length(posterior_weights)
            posterior_weights = fill(1.0 / n, n)
        end
        
        @debug "Belief updated" prior_entropy=entropy(prior_weights) posterior_entropy=entropy(posterior_weights)
        
        return BeliefState(prior.states, posterior_weights)
    catch e
        @error "Belief update failed" exception=(e, catch_backtrace())
        # Safe fallback: uniform belief
        n = length(prior.states)
        return BeliefState(prior.states, fill(1.0/n, n))
    end
end

"""
    observation_likelihood(observation::Observation, state::POMDPState) -> Float64

Compute likelihood P(observation | state).

# Arguments
- `observation::Observation`: Observed data (noisy price)
- `state::POMDPState`: Hypothetical true state

# Returns
- `Float64`: Likelihood (probability density)

# Algorithm
Gaussian observation model:
```
P(obs | state) = N(obs; state.price, σ²)
```
where σ is observation noise standard deviation.

Mathematical form:
```
P(obs | state) = (1 / (σ√(2π))) * exp(-0.5 * ((obs - state.price) / σ)²)
```

# Notes
- Assumes Gaussian noise in price observations
- σ = 0.005 (0.5% noise) - could be parameterized
- Higher likelihood when observed price is close to state price
- Lower likelihood when observed price is far from state price

# Example
```julia
obs = Observation(102.0, 100.0, 1)
state = POMDPState(100.0, 0.0001, 0.02, 1)
likelihood = observation_likelihood(obs, state)
# likelihood is high if obs.observed_price ≈ state.price
# likelihood is low if obs.observed_price far from state.price
```
"""
function observation_likelihood(observation::Observation, state::POMDPState)::Float64
    # Observation noise standard deviation (could be parameterized)
    σ = 0.005  # 0.5% noise
    
    # Difference between observed and true price
    diff = observation.observed_price - state.price
    
    # Gaussian likelihood formula:
    # P(obs|state) = (1/(σ√(2π))) * exp(-0.5 * (diff/σ)²)
    # 
    # Components:
    # - 1/(σ√(2π)): Normalization constant
    # - exp(-0.5 * (diff/σ)²): Gaussian kernel (bell curve)
    # - diff/σ: Standardized difference (z-score)
    
    likelihood = exp(-0.5 * (diff / σ)^2) / (σ * sqrt(2π))
    
    return likelihood
end

"""
    normalize_belief(belief::BeliefState) -> BeliefState

Normalize belief state to ensure valid probability distribution.

# Arguments
- `belief::BeliefState`: Belief state (possibly unnormalized)

# Returns
- `BeliefState`: Normalized belief state

# Algorithm
```
normalized_weights = weights / sum(weights)
```

Special cases:
- If sum = 0: Use uniform distribution (all states equally likely)
- If sum ≈ 1: Already normalized (no change)
- Otherwise: Divide by sum to normalize

# Notes
- Ensures sum(weights) = 1.0 (within numerical tolerance)
- Handles edge case of all-zero weights
- Validates all weights are non-negative
- Creates new BeliefState (doesn't modify input)

# Error Handling
- Checks for negative weights
- Handles zero-sum case gracefully
- Logs normalization at debug level
- Returns uniform belief on error

# Example
```julia
belief = BeliefState(states, [0.6, 0.8, 0.4])  # Sum = 1.8
normalized = normalize_belief(belief)
# normalized.weights = [0.333, 0.444, 0.222]  # Sum = 1.0
```
"""
function normalize_belief(belief::BeliefState)::BeliefState
    try
        # Check for negative weights
        if any(w < 0 for w in belief.weights)
            @warn "Negative weights detected in belief" weights=belief.weights
            # Clip to zero
            weights = max.(belief.weights, 0.0)
        else
            weights = belief.weights
        end
        
        # Compute sum
        total = sum(weights)
        
        if total ≈ 0
            # All weights zero: use uniform distribution
            @warn "All belief weights zero, using uniform distribution"
            n = length(weights)
            normalized_weights = fill(1.0 / n, n)
        elseif isapprox(total, 1.0, atol=1e-6)
            # Already normalized
            normalized_weights = weights
        else
            # Normalize
            normalized_weights = weights ./ total
        end
        
        @debug "Belief normalized" original_sum=total normalized_sum=sum(normalized_weights)
        
        return BeliefState(belief.states, normalized_weights)
    catch e
        @error "Belief normalization failed" exception=(e, catch_backtrace())
        # Safe fallback: uniform belief
        n = length(belief.states)
        return BeliefState(belief.states, fill(1.0/n, n))
    end
end

"""
    uniform_belief(states::Vector{POMDPState}) -> BeliefState

Create uniform belief over given states.

# Arguments
- `states::Vector{POMDPState}`: Possible states

# Returns
- `BeliefState`: Uniform belief (all states equally likely)

# Notes
- Each state has probability 1/n where n = number of states
- Useful for initialization (no prior knowledge)
- Represents maximum uncertainty

# Example
```julia
states = [POMDPState(100.0, 0.0, 0.02, 0) for _ in 1:10]
belief = uniform_belief(states)
# belief.weights = [0.1, 0.1, ..., 0.1]  # All equal
```
"""
function uniform_belief(states::Vector{POMDPState})::BeliefState
    n = length(states)
    weights = fill(1.0 / n, n)
    return BeliefState(states, weights)
end

"""
    entropy(weights::Vector{Float64}) -> Float64

Compute Shannon entropy of probability distribution.

# Arguments
- `weights::Vector{Float64}`: Probability distribution

# Returns
- `Float64`: Entropy in nats (natural logarithm)

# Formula
```
H(p) = -Σ p_i * log(p_i)
```

# Notes
- Measures uncertainty in distribution
- Higher entropy = more uncertain
- Lower entropy = more certain
- Uniform distribution has maximum entropy
- Deterministic distribution (one weight = 1) has zero entropy

# Example
```julia
uniform = [0.25, 0.25, 0.25, 0.25]
entropy(uniform)  # ≈ 1.386 (high uncertainty)

certain = [1.0, 0.0, 0.0, 0.0]
entropy(certain)  # = 0.0 (no uncertainty)
```
"""
function entropy(weights::Vector{Float64})::Float64
    # Filter out zero weights (0 * log(0) = 0 by convention)
    nonzero_weights = filter(w -> w > 0, weights)
    
    # Compute entropy: H = -Σ p * log(p)
    return -sum(w * log(w) for w in nonzero_weights)
end

"""
    discretize_price(price::Float64, min_price::Float64, max_price::Float64, n_bins::Int) -> Int

Discretize continuous price into discrete bin.

# Arguments
- `price::Float64`: Continuous price
- `min_price::Float64`: Minimum price (bin 1)
- `max_price::Float64`: Maximum price (bin n_bins)
- `n_bins::Int`: Number of bins

# Returns
- `Int`: Bin index (1 to n_bins)

# Notes
- Used for Q-learning (discrete state space)
- Linear binning: equal-width bins
- Clips to [min_price, max_price] range
- Returns 1 if price < min_price
- Returns n_bins if price > max_price

# Example
```julia
bin = discretize_price(105.0, 90.0, 110.0, 10)
# bin ∈ {1, 2, ..., 10}
# bin = 8 if price = 105.0 (in 8th bin)
```
"""
function discretize_price(price::Float64, min_price::Float64, max_price::Float64, n_bins::Int)::Int
    # Clip to range
    clipped_price = clamp(price, min_price, max_price)
    
    # Compute bin width
    bin_width = (max_price - min_price) / n_bins
    
    # Compute bin index (1-indexed)
    bin = Int(ceil((clipped_price - min_price) / bin_width))
    
    # Ensure in valid range [1, n_bins]
    return clamp(bin, 1, n_bins)
end

# Exports
export update_belief, observation_likelihood, normalize_belief
export uniform_belief, entropy, discretize_price
