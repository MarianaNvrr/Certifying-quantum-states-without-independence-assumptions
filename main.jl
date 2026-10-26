using Random
using LinearAlgebra
using Ket
using StatsBase
using SparseArrays
using ArnoldiMethod

⊗(x,y) = kron(x,y)

"""
Return a pure n-qubit density matrix for a predefined target state.
"""
function quantum_state(type,n; α=3/4)
    d = 2^n
    if type == "haar"
        ψ = randn(ComplexF64, d)
    elseif type =="weak_entangled"
        ψ = sqrt(α)*ket(1,d) + sqrt(1-α)*ket(d,d)
    elseif type == "ghz" || (type == "phi+" && n == 2)
        ψ = (ket(1,d) + ket(d,d))/sqrt(2)
    elseif type == "w" || (type == "psi-" && n == 2)
        ψ = sum([ket(2^i+1, d) for i=0:n-1])/sqrt(n)
    else
        @error("State not implemented")
    end
    ρ = ψ*ψ'
    ρ ./= real(tr(ψ*ψ'))
    ρ = (ρ+ρ')/2
    return ρ
end

""" 
Return all non-identity `n`-qubit Pauli indices.
"""
function pauli_indices(n)
    pauli_indices = Iterators.product(fill(0:3, n)...) |> collect
    identity_ind = Tuple(zeros(Int, n))
    relevant_indices = filter(x -> x != identity_ind, pauli_indices)
    return relevant_indices
end

"""Return expectation value according to the state type."""
expectation_value(O, state::AbstractVector) = real(state' * (O * state))
expectation_value(O, state::AbstractMatrix) = real(tr(O * state))

##############################################################
####################  SOURCE MODEL ###########################
##############################################################

"""
Generate one step of the coherent drift source model.
Returns the drifted state and the updated angle.
"""
function drifting_coherent_state(rng,target, H_drift, theta_current, drift_rate)
    # Random walk for the rotation angle
    step = drift_rate * randn(rng)
    theta_next = theta_current + step
    
    # Calculate the drifting unitary error
    U_drift = exp(-1im * theta_next * H_drift)
    
    # Apply the coherent error to the target state
    sigma_t = U_drift * target * U_drift'
    sigma_t = (sigma_t + sigma_t') / 2
    
    return sigma_t / real(tr(sigma_t)), theta_next
end


"""
Generate a non-i.i.d. trajectory of drifted quantum states.

Input: 
- `N`: number of states in the trajectory. 
- `target`: target density matrix or vector. 
- (optional) `rate` : standard deviation of each random-walk step.
- (optional) `seed` : defaults to nothing for a fresh random source each call.
- (optional) `stream`: if true, generate states on demand instead of storing them.

Output:
- With `stream=false`: a vector containing N states.
- With `stream=true`: a single-pass iterator over N states.

"""
function generate_drift_trajectory(N, target; rate=0.001, seed=nothing, stream=false) 
    d = size(target, 1)
    N = Int(N)
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)
    theta_t = 0.0

    if target isa AbstractVector
        A = randn(rng, ComplexF64, d, d)
        drift = eigen(Hermitian((A + A') / 2))
        coefficients = drift.vectors' * target

        trajectory = (begin
            theta_t += rate * randn(rng)
            drift.vectors * (exp.(-im * theta_t .* drift.values) .* coefficients)
        end for _ in 1:N)

    else
        H = randn(rng, ComplexF64, d, d)
        H_drift =(H + H') / 2
        trajectory = Vector{Matrix{ComplexF64}}(undef, N)
        
        for t in 1:N
            ρ_t, theta_t = drifting_coherent_state(rng,target, H_drift, theta_t, rate)
            trajectory[t] = ρ_t
        end
    end
    return stream ? trajectory : collect(trajectory)
end


##############################################################
##############  Hamiltonian example functions ################
##############################################################

"""Put a one-qubit Pauli matrix at `site` and identities elsewhere."""
function local_pauli(n, site, pauli_label)
    operators = [sparse(Matrix{ComplexF64}(I, 2, 2)) for _ in 1:n]
    operators[site] = sparse(pauli(pauli_label))
    return foldl(kron, operators)          # kron of sparse stays sparse
end

"""Magnetization m_P = (1/n) sum_i P_i."""
function magnetization_operator(n, pauli_label)
    M = spzeros(ComplexF64, 2^n, 2^n)      # was zeros(...)
    for i in 1:n
        M += local_pauli(n, i, pauli_label) / n
    end
    return M
end


"""Spatially averaged correlation C_PP(r)."""
function correlation_operator(n, r, pauli_label)
    C = zeros(ComplexF64, 2^n, 2^n)
    for i in 1:(n - r)
        C += local_pauli(n, i, pauli_label) *
             local_pauli(n, i + r, pauli_label) 
    end
    return C / (n - r)
end

# The code below is inspired by the tutorial 
# https://juliaphysics.github.io/PhysicsTutorials.jl/tutorials/general/quantum_ising/quantum_ising.html
# by Carsten Bauer and Katharine Hyatt

"""
Compute the smallest and largest eigenvalues and corresponding eigenvectors of a 
sparse Hermitian matrix.
"""
function eigen_sparse(x)
    decomp_min, _ = partialschur(x, nev=1, which=ArnoldiMethod.SR()); # only solve for the ground state
    vals_min, vecs_min = partialeigen(decomp_min);

    decomp_max, _ = partialschur(x, nev=1, which=ArnoldiMethod.LR())
    vals_max, vecs_max = partialeigen(decomp_max)

    vals = real.([vals_min, vals_max])
    vecs = [vecs_min, vecs_max]

    return vals, vecs
end

"""
Construct the `n`-qubit transverse-field Ising Hamiltonian.
The Hamiltonian is returned as a sparse matrix.
"""
function TFIM_hamiltonian(n;J=1.0, g=0.5)
    id = [1 0; 0 1] |> sparse
    σˣ = [0 1; 1 0] |> sparse
    σᶻ = [1 0; 0 -1] |> sparse
    
    first_term_ops = fill(id, n)
    first_term_ops[1] = σᶻ
    first_term_ops[2] = σᶻ
    
    second_term_ops = fill(id, n)
    second_term_ops[1] = σˣ
    
    H = spzeros(Int, 2^n, 2^n) # note the spzeros instead of zeros here
    for _ in 1:n-1
        H -= foldl(⊗, first_term_ops)
        first_term_ops = circshift(first_term_ops,1)
    end
    
    for _ in 1:n
        H -= g*foldl(⊗, second_term_ops)
        second_term_ops = circshift(second_term_ops,1)
    end

    return J*H
end

"""
Return the ground-state density matrix of a Hamiltonian `H`.
Set `density_matrix=false` to return the corresponding state vector instead.
"""
function ground_state(H; density_matrix=true)
    # Diagonalize to find the ground state
    _ , vecs = eigen_sparse(H)
    
    #eigenvector for the lowest energy eigenvalue
    psi_gs = vecs[1][:, 1] 
    
    return density_matrix ? psi_gs * psi_gs' : psi_gs
end


##############################################################
########### Entanglement witness example functions ###########
##############################################################

"""The 24 single-qubit Clifford unitaries, generated by H and S, modulo a global phase."""
function single_qubit_cliffords()
    H = ComplexF64[1 1; 1 -1] / sqrt(2)
    S = ComplexF64[1 0; 0 im]

    # Fix the global phase: the first nonzero entry is made real and positive
    canonical(U) = (k = findfirst(x -> abs(x) > 1e-9, U); U .* (abs(U[k]) / U[k]))

    cliffords = [Matrix{ComplexF64}(I, 2, 2)]
    frontier = copy(cliffords)
    while !isempty(frontier)
        new_elements = Matrix{ComplexF64}[]
        for U in frontier, G in (H, S)
            V = canonical(G * U)
            if !any(C -> isapprox(C, V; atol=1e-9), cliffords)
                push!(cliffords, V)
                push!(new_elements, V)
            end
        end
        frontier = new_elements
    end

    return cliffords
end

"""
Family of witnesses obtained by conjugating an n-qubit witness `W` with local
Cliffords, O = (C_1 ⊗ ... ⊗ C_n) W (C_1 ⊗ ... ⊗ C_n)†. Duplicates are removed
and `W` is the first element.

Local unitaries map product states to product states, so every member is again
a witness. Local Cliffords map Pauli strings to Pauli strings on the same qubits,
so every member also has the same shadow norm for local Pauli measurements.
"""
function local_clifford_family(W)
    n = Int(log2(size(W, 1)))
    cliffords = single_qubit_cliffords()
    family = [Matrix{ComplexF64}(W)]

    for local_cliffords in Iterators.product(fill(cliffords, n)...)
        U = foldl(kron, local_cliffords)
        O = U * W * U'
        O = (O + O') / 2
        any(M -> isapprox(M, O; atol=1e-9), family) || push!(family, O)
    end

    return family
end


##############################################################
####################  Classical Shadows ######################
##############################################################

# U * rho * U' rotates an X-, Y-, or Z-basis measurement into the
# computational basis. The rows are the corresponding Pauli eigenvectors.
const BASIS_ROTATIONS = ([1 1; 1 -1] / sqrt(2), [1 -im; 1 im] / sqrt(2), I(2))


"""Apply one 2x2 matrix to one qubit of a state vector."""
function apply_one_qubit!(output, input, U, site, n)
    shift = n - site
    bit_mask = 1 << shift

    @inbounds for basis_state in 0:(length(input) - 1)
        # Treat each pair only once: the member whose selected qubit is zero.
        if ((basis_state >> shift) & 1) == 0
            paired_state = xor(basis_state, bit_mask)
            amplitude_0 = input[basis_state + 1]
            amplitude_1 = input[paired_state + 1]

            output[basis_state + 1] = U[1, 1] * amplitude_0 + U[1, 2] * amplitude_1
            output[paired_state + 1] = U[2, 1] * amplitude_0 + U[2, 2] * amplitude_1
        end
    end
end


"""Rotate each qubit locally without constructing a global 2^n x 2^n matrix."""
function rotate_local_bases(state, bases, n)
    rotated = ComplexF64.(state)
    buffer = similar(rotated)
    
    for i in 1:n
        apply_one_qubit!(buffer, rotated, BASIS_ROTATIONS[bases[i]], i, n)
        rotated, buffer = buffer, rotated
    end

    return rotated
end


"""Decompose a traceless matrix into `(index, labels, coefficient)` terms."""
function pauli_decomposition(W, n; tolerance=1e-12)
    d = 2^n
    terms = Tuple{Vector{Int},Vector{Int},Float64}[]

    for index in pauli_indices(n)
        P = pauli(collect(index))
        alpha_P = real(tr(W * P)) / d
        abs(alpha_P) <= tolerance && continue

        labels_at_all_index = collect(index)
        index = findall(!=(0), labels_at_all_index)
        labels = labels_at_all_index[index]
        push!(terms, (index, labels, alpha_P))
    end

    return terms
end


"""
Construct the exact second-moment operator `Q_W` for uniformly random
local-Pauli measurements.

The input uses the same sparse Pauli representation as
`shadow_observable_estimator`: every entry is `(sites, labels, coefficient)`,
where labels 1, 2, and 3 denote X, Y, and Z. Compatible pairs are collected by
their Pauli product before the corresponding matrices are constructed.
"""
function second_moment_exact(terms, n)
    n_terms = length(terms)

    # Convert each sparse representation into a full Pauli-label string.
    pauli_strings = [zeros(Int, n) for _ in 1:n_terms]
    coefficients = zeros(Float64, n_terms)

    for (j, (sites, labels, coefficient)) in enumerate(terms)
        pauli_strings[j][sites] .= labels
        coefficients[j] = coefficient
    end

    # Several compatible pairs can give the same product P*P'. Aggregate their
    # coefficients before constructing any exponentially large matrices.
    identity_key = Tuple(zeros(Int, n))
    product_coefficients = Dict{typeof(identity_key),Float64}()

    for a in 1:n_terms
        P = pauli_strings[a]

        # The analytical formula sums over ordered pairs. We visit only the
        # upper triangle and double off-diagonal contributions.
        for b in a:n_terms
            P_prime = pauli_strings[b]
            compatible = true
            overlap = 0
            product_labels = zeros(Int, n)

            for i in 1:n
                label_P = P[i]
                label_P_prime = P_prime[i]

                if label_P != 0 && label_P_prime != 0
                    if label_P != label_P_prime
                        compatible = false
                        break
                    end

                    # we have a contribution when the paulis are equal or 
                    # at least one is identity
                    overlap += 1
                else
                    # At most one of the strings acts nontrivially at this site
                    product_labels[i] = label_P == 0 ? label_P_prime : label_P
                end
            end

            # An incompatible pair contributes zero
            compatible || continue

            multiplicity = a == b ? 1.0 : 2.0
            pair_coefficient = multiplicity * coefficients[a] * coefficients[b] * 3.0^overlap
            product_key = Tuple(product_labels)
            product_coefficients[product_key] = get(product_coefficients, product_key, 0.0) + pair_coefficient
        end
    end

    d = 2^n
    Q_W = zeros(ComplexF64, d, d)

    for (product_labels, coefficient) in product_coefficients
        Q_W .+= coefficient .* pauli(collect(product_labels))
    end

    # ensure no floating-point prevents hermicity
    return Hermitian((Q_W + Q_W') / 2)
end


"""
Perform one random local-Pauli measurement on a state vector or density matrix.

Returns the compact shadow record `(bases, outcomes)`, not the snapshot `rho_hat`.
All `2^n` joint Born probabilities are obtained from one basis rotation.
"""
function measure_classical_shadow(state, n; rng=Random.default_rng())
    # Choose a random Pauli basis for every qubit
    # 1 -> X, 2 -> Y, and 3 -> Z.
    bases = rand(rng, 1:3, n)

    if state isa AbstractVector
        # rotates the state in the corresponding basis and 
        # compute its probability
        ρ_rotated = rotate_local_bases(state, bases, n)
        probabilities = abs2.(ρ_rotated)
    else
        local_rotations = [BASIS_ROTATIONS[bases[i]] for i in 1:n]
        joint_rotation = foldl(kron, local_rotations)
        ρ_rotated = joint_rotation * state * joint_rotation'
        probabilities = max.(real.(diag(ρ_rotated)), 0.0)
    end
    probabilities ./= sum(probabilities)

    # Sample one computational-basis bit string
    outcome_sample = sample(rng, 0:(2^n - 1), pweights(probabilities))

    # Decode the integer into n bits, ordered from qubit 1 to qubit n.
    bits = reverse(digits(outcome_sample; base=2, pad=n))

    # Convert bits into Pauli eigenvalues.
    measured_outcomes = [bit == 0 ? +1 : -1 for bit in bits]
    return (bases=bases, outcomes=measured_outcomes)
end


"""Evaluate `tr(W*rho_hat)` directly from one compact shadow record"""
function shadow_observable_estimator(bases, outcomes, terms)
    estimate = 0.0

    for (sites, labels, coefficient) in terms
        # A Pauli term contributes only when every required basis matches.
        # It correspond to the kronecker delta between b and P (see Eq. (A3))
        all(bases[site] == label for (site, label) in zip(sites, labels)) || continue

        measured_sign = prod(outcomes[site] for site in sites)
        estimate += coefficient * 3.0^length(sites) * measured_sign
    end

    return estimate
end
