include("main.jl")

"""
Run the importance-sampled Pauli protocol for a Hamiltonian observable `H`.
The protocol estimates the average energy of the use states
using only the randomly selected test states.

Input
- `N`: total number of experimental rounds.
- `H`: Hamiltonian observable, assumed to be an n-qubit Hermitian matrix.
- (optional) `p`: probability of testing a state.
- (optional) `δ`: failure probability.
- (optional) `drift_rate`: how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Output:
- `radius`: certification radius 
- `estimate`: estimate of the average energy of the use states, inferred from test rounds.
- `true_use_average`: exact average energy of the use states,
  computed only for numerical validation.

The source drifts from the ground state of `H`. The radius uses Theorem 4
and Eq. (51), and requires at least one use round.
"""
function hamiltonian_pauli_IS(N,H; p=0.01, δ=0.05,drift_rate=0.001,verbose=true)
    d = size(H, 1); n = Int(log2(d))
    N = Int(N)
    q = 1 - p

    # Non-identity Pauli labels.
    pauli_labels = pauli_indices(n)

    # Pauli coefficients α_P, Eq. (39).
    α_P = Float64[]
    for pauli_label in pauli_labels
        P = pauli(collect(pauli_label))
        push!(α_P, real(tr(H * P)) / d)
    end

    L1 = sum(abs.(α_P))
    π_P = pweights(abs.(α_P) / L1)

    # Freedman bound parameters R and v, Eqs. (49)-(50).
    values = eigen_sparse(H)[1]
    W_min = values[1][1]
    W_max = values[2][1]
    W_inf = max(abs(W_min), abs(W_max))

    R = max(W_inf, (q/p) * L1)
    v = q * W_inf^2 + (q^2/p) * L1^2

    use_values = Float64[]      # Exact values on use rounds (validation only).
    test_estimators = Float64[] # Single-shot estimates on test rounds.

    K = p * N
    threshold_term = R/3 * log(2/δ)
    ϵ = (threshold_term + sqrt(2*N*v*log(2/δ) + threshold_term^2)) / (q*N)

    if verbose
        println("--- PRIOR BOUND SETUP ---")
        println("Total rounds N=$N, expected test rounds K≈$(round(K, digits=0)), target ϵ≈$ϵ (Theorem 4) \n")
    end

    # Initialize the drifting source at the Hamiltonian ground state.
    target = ground_state(H; density_matrix=false)
    trajectory = generate_drift_trajectory(N, target; rate=drift_rate, stream=true)

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

                # Single-shot estimator for the traceless part only, Eq. (42)
                X_t = y_t * L1 * sign(α_P[sample_idx])
                push!(test_estimators, X_t)
            else # Use round (numerical validation only).
                ω_t = expectation_value(H, ρ_t)
                push!(use_values, ω_t)
            end
        end
    end

    num_use = length(use_values)
    num_test = length(test_estimators)

    estimate = q/p * (sum(test_estimators)/num_use)

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
        println("TRUE Avg energy of use states     :    = $(round(true_use_average, digits=5))")
        println("Certified Lower Bound             :    ≥ $(round(certified_lower_bound, digits=5))")
        println("Empirical Confidence Width        :    ± $(round(radius, digits=5))")
        println("TRUE Avg energy of use states     :    = $(round(true_use_average, digits=5))")
        println("_________________________________________________")

        if true_use_average >= certified_lower_bound
            println("\n SUCCESS: True average energy is above the certified lower bound. \n")
        else
            println("\n Outside the lower bound in this run (allowed with probability <= delta). \n")
        end
    end

    return radius, estimate, true_use_average
end


"""
Run the spot-checking classical shadows protocol for various observables.

Input:
- `N`: total number of experimental rounds.
- `observables`: named tuple of n-qubit Hermitian observables. The first observable
  is the Hamiltonian whose ground state initializes the source.
- (optional) `p`: probability of testing a state.
- (optional) `δ`: simultaneous failure probability.
- (optional) `drift_rate`: how much the state randomly drifts at each step.
- (optional) `verbose`: if true, print protocol diagnostics and results.

Output:
- `radius`: common certification radius for all average observable values (Theorem 5 and Eq. (63)).
- `estimates`: estimates of the average observable values of the use states from the same test data
- `true_use_averages`: exact average observable values of the use states,
  computed only for numerical validation.

"""
function hamiltonian_shadows(N, observables; p=0.1, δ=0.05, drift_rate=0.001, verbose=true)
    H = observables[1]
    d = size(H, 1)
    n = Int(log2(d))
    N = Int(N)
    q = 1 - p
    L = length(observables)
    observable_names = string.(collect(keys(observables)))
    observables = collect(values(observables))

    α_I = [real(tr(O)) / d for O in observables]
    traceless_observables = [O - α_I[j] * I(d) for (j, O) in enumerate(observables)]

    pauli_terms = [pauli_decomposition(W, n) for W in traceless_observables]

    # Exact squared shadow norms, Appendix A / Eq. (A6)
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
        for j in eachindex(observable_names)
            println("  $(observable_names[j]): exact squared shadow norm = " *
                    "$(round(shadow_norms_squared[j], digits=5))")
        end
        println("Threshold T = $(round(T, digits=5))\n")
    end

    # Initialize the drifting source at the Hamiltonian ground state.
    target = ground_state(H; density_matrix=false)
    trajectory = generate_drift_trajectory(N, target; rate=drift_rate, stream=true)

    # One sum for each observable.
    sum_test_estimators = zeros(L) # Single-shot estimates on test rounds.
    sum_true_use_values = zeros(L) # Exact values on use rounds (validation only).
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
                # directly to the quantum task and are not measured here.
                for j in 1:L
                    o_tj = expectation_value(observables[j], ρ_t)
                    sum_true_use_values[j] += o_tj
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
        for j in 1:L
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
    end

    return radius, estimates, true_use_averages
end
