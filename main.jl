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
function state(type,n; α=3/4)
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

#-------------- SOURCE MODEL ----------------------

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
- `target`: target density matrix. 
- (optional) `rate`: standard deviation of each random-walk step. 
- (optional) `seed`: random seed for reproducibility.

Output: 
- vector of density matrices `[ρ_1, ..., ρ_N]`.
"""
function generate_drift_trajectory(N, target::Matrix{T}; rate=0.001, seed=42) where {T<:Number}
    d = size(target, 1)
    
    N = Int(N)

    rng = MersenneTwister(seed)
    H = randn(rng, ComplexF64, d, d)
    H_drift =(H + H') / 2

    theta_t = 0.0
    
    trajectory = Vector{Matrix{ComplexF64}}(undef, N)
    
    for t in 1:N
        ρ_t, theta_t = drifting_coherent_state(rng,target, H_drift, theta_t, rate)
        trajectory[t] = ρ_t
    end
    
    return trajectory
end


##############################################################
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
Construct the `n`-qubit XXZ Hamiltonian.
The Hamiltonian is returned as a sparse matrix.
"""
function XXZ_hamiltonian(n;J=1.0, Δ=1.)
    id = sparse(ComplexF64[1 0; 0 1])
    σˣ = sparse(ComplexF64[0 1; 1 0])
    σʸ = sparse(ComplexF64[0 -im; im 0])  
    σᶻ = sparse(ComplexF64[1 0; 0 -1])
    
    XX_ops = fill(id, n)
    XX_ops[1], XX_ops[2] = σˣ, σˣ
    
    YY_ops = fill(id, n)
    YY_ops[1], YY_ops[2] = σʸ, σʸ
    
    ZZ_ops = fill(id, n)
    ZZ_ops[1], ZZ_ops[2] = σᶻ, σᶻ
    
    second_term_ops = fill(id, n)
    second_term_ops[1] = σˣ
    
    H = spzeros(Int, 2^n, 2^n) # note the spzeros instead of zeros here
    for _ in 1:n-1
        H += foldl(⊗, XX_ops)
        H += foldl(⊗, YY_ops)
        H += Δ * foldl(⊗, ZZ_ops)
        
        # Shift all operators one site over
        XX_ops = circshift(XX_ops, 1)
        YY_ops = circshift(YY_ops, 1)
        ZZ_ops = circshift(ZZ_ops, 1)
    end

    return J*H
end

"""
Return the ground-state density matrix of a Hamiltonian `H`.
"""
function ground_state(H)
    # Diagonalize to find the ground state
    _ , vecs = eigen_sparse(H)
    
    #eigenvector for the lowest energy eigenvalue
    psi_gs = vecs[1][:, 1] 
    
    return psi_gs * psi_gs'
end

