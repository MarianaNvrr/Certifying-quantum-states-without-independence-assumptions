include("main.jl")

"""
Run the certification protocol for an entanglement witness `W`, using
importance-sampled Pauli measurements on the test rounds.
The protocol estimates the average witness value of the use states
using only the randomly selected test states.

Input
- `st`: target state used to generate the simulated drifting trajectory,
  given as a state vector or a density matrix.
- `N`: total number of experimental rounds.
- `W`: witness to certify, assumed to be an n-qubit Hermitian matrix.
  It does not need to be traceless.
- (optional) `p`: probability of testing a state.
- (optional) `δ`: failure probability.
- (optional) `drift_rate`: controls how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Output
- `radius`: certification radius (see Theorem 4).
- `estimate`: estimator inferred from the test rounds.
- `true_use_average`: exact average witness value of the use states,
  computed only for numerical validation.
"""
function witness_Pauli_IS(st,N,W; p=0.077, δ=0.05, drift_rate=0.001, verbose=true)
    d = size(W, 1); n = Int(log2(d))
    N = Int(N)
    q = 1 - p

    # Non-identity Pauli labels.
    pauli_labels = pauli_indices(n)

    α_P = Float64[]; α_I = real(tr(W))/d
    for pauli_label in pauli_labels
        P = pauli(collect(pauli_label))
        push!(α_P, real(tr(W * P))/d)
    end

    L1 = sum(abs.(α_P))
    π_P = pweights(abs.(α_P)/(L1) )

    # Freedman (variance-sensitive) bound parameters, Theorem 4 / Eqs. (49)-(50)
    # Note: this uses the traceless part of W only
    W_traceless = Hermitian(Matrix(W - α_I * I(d)))
    W_min, W_max = extrema(eigvals(W_traceless))
    W_inf = max(abs(W_min), abs(W_max))

    R = max(W_inf, (q/p) * L1)
    v = q * W_inf^2 + (q^2/p) * L1^2

    use_values = Float64[] # Exact values on use rounds (validation only).
    test_estimators     = Float64[] # Single-shot estimates on test rounds.

    K = p * N
    threshold_term = R/3 * log(2/δ)
    ϵ = (threshold_term + sqrt(2*N*v*log(2/δ) + threshold_term^2)) / (q*N)

    if verbose
        println("--- PRIOR BOUND SETUP ---")
        println("Total rounds N=$N, expected test rounds K≈$(round(K, digits=0)), target ϵ≈$ϵ (Theorem 4) \n")
    end

    # Source model. We need to assume a state for the simulations.
    trajectory = generate_drift_trajectory(N, st; rate=drift_rate, stream=true)

    # MAIN LOOP
    loop_time = @elapsed begin
        for ρ_t in trajectory
            # Private test/use coin c_t.
            c_t = rand() < p ? 1 : 0

            if c_t == 1 # Test round.
                # We sample from the distribution π_P
                sample_idx = sample(1:length(pauli_labels), π_P)
                pauli_label = collect(pauli_labels[sample_idx])
                P_t = pauli(pauli_label)

                p_plus = (1 + expectation_value(P_t, ρ_t)) / 2
                y_t = rand() < p_plus ? 1.0 : -1.0

                # Single-shot estimator for the traceless part only
                X_t = y_t * L1 * sign(α_P[sample_idx])
                push!(test_estimators, X_t)
            else # Use round (numerical validation only).
                true_val = expectation_value(W, ρ_t)
                push!(use_values, true_val)
            end
        end
    end

    num_use = length(use_values)
    num_test = length(test_estimators)

    estimate = α_I + q/p * (sum(test_estimators)/num_use)

    # Freedman-based certification radius, Theorem 4 / Eq. (51)
    radius = (threshold_term + sqrt(2*N*v*log(2/δ) + threshold_term^2)) / num_use

    certified_lower_bound = estimate - radius
    certified_upper_bound = estimate + radius
    true_use_average = sum(use_values) / num_use

    if verbose
        println("--- PROTOCOL RESULTS ---")
        println("States Tested (|S_T| $(p*100)%) : $num_test")
        println("States Used (|S_U| $((1-p)*100)%):   $num_use")
        println("The certification took $(round(loop_time, digits=2)) seconds.")
        println("-------------------------------------------------")
        println("Certified Upper Bound             :    ≤ $(round(certified_upper_bound, digits=5))")
        println("TRUE Avg witness of use states    :    = $(round(true_use_average, digits=5))")
        println("Certified Lower Bound             :    ≥ $(round(certified_lower_bound, digits=5))")
        println("Empirical Confidence Width        :    ± $(round(radius, digits=5))")
        println("_________________________________________________")

        if abs(true_use_average - estimate) <= radius
            println("\n SUCCESS: The true witness lies inside the certified interval.")
        else
            println("\n FAILURE: The true witness fell outside the interval (allowed with probability <= δ).")
        end

        # A negative upper bound certifies the entanglement of the use states
        if certified_upper_bound < 0
            println(" Entanglement of the use states is certified.\n")
        else
            println(" Entanglement of the use states is NOT certified.\n")
        end
    end

    return radius, estimate, true_use_average
end

"""
Computes the sample complexity according to Theorem 3 and runs the
verification protocol with importance-sampled Pauli measurements.

Input
- `trajectory`: sequence of states ρ_t (state vectors or density matrices).
  Only its first states are read, so a stateful stream is left positioned
  at the first state after the verification block.
- `W`: observable to estimate.
- `ϵ`: target additive error.
- `δ`: failure probability.

Output
- Estimated average value of `W`.
- Number of samples used by the verification protocol.
"""
function complexity_verification_Pauli(trajectory,W, ϵ, δ)
    d = size(W, 1); n = Int(log2(d))

    # Non-identity Pauli labels.
    pauli_labels = pauli_indices(n)

    # Pauli coefficients α_P
    α_P = Float64[]
    for pauli_label in pauli_labels
        P = pauli(collect(pauli_label))
        push!(α_P, real(tr(W * P))/d)
    end

    L1 = sum(abs.(α_P) )
    π_P = pweights(abs.(α_P)/L1)

    # Sample complexity from Theorem 3
    N = Int(ceil(2 * (L1 / ϵ)^2 * log(2 / δ)))

    X_hat_sum=0.

    # MAIN LOOP
    for ρ_t in Iterators.take(trajectory, N)
        sample_idx = sample(1:length(pauli_labels), π_P)
        pauli_label = collect(pauli_labels[sample_idx])
        P_t = pauli(pauli_label)

        p_plus = (1 + expectation_value(P_t, ρ_t)) / 2
        y_t = rand() < p_plus ? 1.0 : -1.0

        # Single-shot estimator, Eq. (42)
        X_t = y_t * L1 * sign(α_P[sample_idx])
        X_hat_sum += X_t
    end

    return (real(tr(W))/d) + (X_hat_sum / N), N
end

"""
Run a static certification protocol for an observable `W`.
The protocol first consumes an initial block of states to
estimate the value of `W` using the verification protocol.
We also estimate the average witness of the remaining states for
benchmarking.

Input
- `st`: target state used to generate the simulated drifting trajectory,
  given as a state vector or a density matrix.
- `N`: total number of rounds.
- `W`: witness to estimate.
- (optional) `ϵ`: target additive error for the initial verification block.
- (optional) `δ`: failure probability for the verification estimate.
- (optional) `drift_rate`: controls how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Output
- `estimate`: estimate obtained from the initial verified block.
- `true_use_average`: exact average value of `W` on the remaining use states,
computed only for numerical validation.
"""
function witness_static_Pauli(st,N, W; ϵ=0.03, δ=0.05, drift_rate=0.001, verbose=true)
    N = Int(N)
    use_values = Float64[]

    # Source model. We need to assume a state for the simulations.
    # The verification block consumes its first N_ver
    # states and the loop below continues with the remaining ones.
    trajectory = Iterators.Stateful(generate_drift_trajectory(N, st; rate=drift_rate, stream=true))

    estimate, N_ver = complexity_verification_Pauli(trajectory,W, ϵ, δ)
    N_rest = N - N_ver

    if verbose
        println("----- VERIFICATION PROTOCOL RESULTS --------")
        println("Verified states   : $N_ver ($(N_ver*100/N)%)")
        println("Used states       : $N_rest ($(N_rest*100/N)%)")
    end

    loop_time = @elapsed begin
        for ρ_t in trajectory
            true_val = expectation_value(W, ρ_t)
            push!(use_values, true_val)
        end
    end

    num_use = length(use_values)

    true_use_average = sum(use_values)/ num_use

    if verbose
        println("The verification took $(round(loop_time, digits=2)) seconds.")
        println("-------------------------------------------")
        println("Verified witness  : $estimate")
        println("True witness      : $true_use_average")
        println("_____________________________________________")
    end

    return estimate, true_use_average
end

"""
Run the spot-checking classical-shadow protocol for a family of observables.
The same random local-Pauli measurements on the test rounds certify the
average value of every observable over the use states simultaneously.

Input
- `st`: target state used to generate the simulated drifting trajectory,
  given as a state vector or a density matrix.
- `N`: total number of experimental rounds.
- `observables`: named tuple or vector of n-qubit Hermitian matrices, e.g.
  `local_clifford_family(W)`. The first entry is taken to be the witness `W`;
  the others are certified alongside it.
- (optional) `p`: probability of testing a state.
- (optional) `δ`: failure probability for the whole family.
- (optional) `drift_rate`: controls how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Since the bound holds for all observables simultaneously, the witness with the
smallest upper bound can be chosen after the data are collected.

Output
- `radius`: common simultaneous certification radius (see Theorem 5).
- `estimates`: estimator of every observable inferred from the test rounds.
- `true_use_averages`: exact average value of every observable on the use
  states, computed only for numerical validation.
"""
function witness_shadows(st,N,observables; p=0.077, δ=0.05, drift_rate=0.001, verbose=true)
    d = size(observables[1], 1); n = Int(log2(d))
    N = Int(N)
    q = 1 - p
    L = length(observables)
    observable_names = string.(collect(keys(observables)))
    observables = collect(values(observables))

    α_I = [real(tr(O)) / d for O in observables]
    traceless_observables = [O - α_I[j] * I(d) for (j, O) in enumerate(observables)]

    pauli_terms = [pauli_decomposition(W, n) for W in traceless_observables]

    # Exact squared shadow norms, Appendix A
    shadow_norms_squared = [eigmax(second_moment_exact(terms, n)) for terms in pauli_terms]
    S2 = maximum(shadow_norms_squared)

    K = p*N
    ϵ = 5 * sqrt((1 + q/6) * S2 * log(4L/δ) / (2q*K))
    T = 5 * S2 / (4 * ϵ)
    R = T * max(1, q / p)

    if verbose
        println("--- PRIOR BOUND SETUP (THEOREM 5) ---")
        println("Total rounds N = $N, expected test rounds pN = $(round(K, digits=0)), target ϵ≈$ϵ")
        println("Number of observables L = $L")
        println("S^2 = $(round(S2, digits=5))")
        if L <= 10
            for j in eachindex(observable_names)
                println("  $(observable_names[j]): exact squared shadow norm = " *
                        "$(round(shadow_norms_squared[j], digits=5))")
            end
        end
        println("Threshold T = $(round(T, digits=5))\n")
    end

    # Source model. We need to assume a state for the simulations.
    # The states are generated on demand, so the trajectory is never stored.
    trajectory = generate_drift_trajectory(N, st; rate=drift_rate, stream=true)

    # One sum for each observable and the sum of the use state-
    sum_test_estimators = zeros(L) # Single-shot estimates on test rounds.
    sum_true_use_values = zeros(L)
    num_test = 0
    num_use = 0

    # MAIN LOOP
    loop_time = @elapsed begin
        for ρ_t in trajectory
            # Private test/use coin c_t.
            c_t = rand() < p ? 1 : 0

            if c_t == 1 # Test round.
                num_test += 1
                k_t = measure_classical_shadow(ρ_t, n)

                # One compact informationally complete record is reused for
                # every observable. No dense rho_hat matrix is constructed.
                for j in 1:L
                    X_tj = shadow_observable_estimator(k_t.bases, k_t.outcomes, pauli_terms[j])

                    # Y_{t,j}: clipped estimator, Eq. (57)
                    Y_tj = clamp(X_tj, -T, T)
                    sum_test_estimators[j] += Y_tj
                end
            else # Use round (numerical validation only).
                num_use += 1
                # Validation only: in a real protocol these states are passed
                # directly to the quantum task 
                for j in 1:L
                    value = expectation_value(observables[j],ρ_t)
                    sum_true_use_values[j] += value
                end    
            end        
        end
    end

    estimates = α_I .+ (q / p) .* sum_test_estimators ./ num_use
    true_use_averages = sum_true_use_values ./ num_use

    # Simultaneous certification radius, Theorem 5 / Eq. (63)
    threshold_term = (R / 3) * log(2 * L / δ)
    radius = ϵ / 5 + threshold_term / num_use +
             sqrt(2N*(q/p) * S2 * log(2 * L / δ) + threshold_term^2) / num_use

    if verbose
        println("--- PROTOCOL RESULTS ---")
        println("States Tested (|S_T| $(p*100)%) : $num_test")
        println("States Used (|S_U| $((1-p)*100)%):   $num_use")
        println("Runtime: $(round(loop_time, digits=2)) seconds")
        println("Common simultaneous radius: +/- $(round(radius, digits=5))")
        println("------------------------------------------------------------")

        # For large families, only the witness W and the best observable are shown
        best = argmin(estimates)
        for j in (L <= 10 ? (1:L) : unique([1, best]))
            error_j = abs(estimates[j] - true_use_averages[j])
            println(
            rpad(observable_names[j], 12),
                " estimate = ", round(estimates[j], digits=5),
                " true use average = ", round(true_use_averages[j], digits=5),
                " error = ", round(error_j, digits=5))
        end
        max_error = maximum(abs.(estimates .- true_use_averages))
        if max_error <= radius
            println("\n SUCCESS: every observable lies inside the simultaneous interval.")
        else
            println("\n Outside the interval in this run (allowed with probability <= delta).")
        end

        # A negative upper bound certifies the entanglement of the use states
        if estimates[1] + radius < 0
            println(" Entanglement of the use states is certified by $(observable_names[1]).")
        else
            println(" Entanglement of the use states is NOT certified by $(observable_names[1]).")
        end
        # Valid after looking at the data, because the bound is simultaneous
        println(" Smallest upper bound: $(observable_names[best]) ≤ $(round(estimates[best] + radius, digits=5))" *
                " (a negative value certifies entanglement if $(observable_names[best]) is a witness)")
        println()
    end

    return radius, estimates, true_use_averages
end



"""
Computes the sample complexity of classical-shadow verification, Eq. (45) of arXiv:2603.05137,
and runs the verification protocol on the first rounds of the trajectory.

Input
- `trajectory`: sequence of states ρ_t (state vectors or density matrices).
  Only its first states are read, so a stateful stream is left positioned
  at the first state after the verification block.
- `observables`: named tuple or vector of observables, e.g. `local_clifford_family(W)`.
- `ϵ`: target additive error, simultaneous for all the observables.
- `δ`: failure probability.

Output
- Estimated average value of every observable.
- Number of samples used by the verification protocol.
"""
function complexity_verification_shadows(trajectory, observables, ϵ, δ)
    d = size(observables[1], 1); n = Int(log2(d))
    observables = collect(values(observables))
    L = length(observables)

    α_I = [real(tr(O)) / d for O in observables]
    traceless_observables = [O - α_I[j] * I(d) for (j, O) in enumerate(observables)]
    pauli_terms = [pauli_decomposition(W, n) for W in traceless_observables]

    # Exact squared shadow norms, Appendix A
    shadow_norms_squared = [eigmax(second_moment_exact(terms, n)) for terms in pauli_terms]
    S2 = maximum(shadow_norms_squared)

    # Sample complexity and truncation thresholds, Eq. (45) of arXiv:2603.05137
    N = Int(ceil(125/24 * S2 / ϵ^2 * log(2L / δ)))
    T = 5 .* shadow_norms_squared ./ (4ϵ)

    sum_estimators = zeros(L)

    # MAIN LOOP
    for ρ_t in Iterators.take(trajectory, N)
        k_t = measure_classical_shadow(ρ_t, n)

        # One shadow record is reused for every observable
        for j in 1:L
            X_tj = shadow_observable_estimator(k_t.bases, k_t.outcomes, pauli_terms[j])
            sum_estimators[j] += clamp(X_tj, -T[j], T[j])
        end
    end

    return α_I .+ sum_estimators ./ N, N
end

"""
Run a static certification protocol with classical shadows for a family of
observables. The protocol first consumes an initial block of states to
estimate every observable with the shadow verification protocol, and assumes
the estimates hold for the remaining states. We also compute the average value
of every observable on the remaining states for benchmarking.

Input
- `st`: target state used to generate the simulated drifting trajectory,
  given as a state vector or a density matrix.
- `N`: total number of rounds.
- `observables`: named tuple or vector of observables, e.g. `local_clifford_family(W)`.
  The first entry is taken to be the witness `W`.
- (optional) `ϵ`: target additive error for the initial verification block,
  simultaneous for all the observables.
- (optional) `δ`: failure probability for the verification estimate.
- (optional) `drift_rate`: controls how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Output
- `estimates`: estimate of every observable from the initial verified block.
- `true_use_averages`: exact average value of every observable on the remaining
  use states, computed only for numerical validation.
"""
function witness_static_shadows(st, N, observables; ϵ=0.04, δ=0.05, drift_rate=0.001, verbose=true)
    N = Int(N)
    observable_names = string.(collect(keys(observables)))
    observables = collect(values(observables))
    L = length(observables)

    # Source model. We need to assume a state for the simulations.
    # The verification block consumes its first N_ver
    # states and the loop below continues with the remaining ones.
    trajectory = Iterators.Stateful(generate_drift_trajectory(N, st; rate=drift_rate, stream=true))

    estimates, N_ver = complexity_verification_shadows(trajectory, observables, ϵ, δ)
    N_rest = N - N_ver

    if verbose
        println("----- SHADOW VERIFICATION PROTOCOL RESULTS --------")
        println("Verified states   : $N_ver ($(N_ver*100/N)%)")
        println("Used states       : $N_rest ($(N_rest*100/N)%)")
    end

    sum_true_use_values = zeros(L)
    num_use = 0

    loop_time = @elapsed begin
        for ρ_t in trajectory
            num_use += 1
            for j in 1:L
                sum_true_use_values[j] += expectation_value(observables[j], ρ_t)
            end
        end
    end

    true_use_averages = sum_true_use_values ./ num_use

    if verbose
        best = argmin(estimates)
        println("The verification took $(round(loop_time, digits=2)) seconds.")
        println("-------------------------------------------")
        for j in unique([1, best])
            println(rpad(observable_names[j], 12),
                " verified = ", round(estimates[j], digits=5),
                " true use average = ", round(true_use_averages[j], digits=5))
        end
        println("_____________________________________________")
    end

    return estimates, true_use_averages
end
