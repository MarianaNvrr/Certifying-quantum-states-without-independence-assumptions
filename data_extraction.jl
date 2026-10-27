using Statistics, JLD2, NPZ, Plots

default(fontfamily = "Computer Modern",
        thickness_scaling = 0.6,
        size = (245, 187),             
        left_margin = 2Plots.mm,
        right_margin = 3Plots.mm,
        top_margin = 2Plots.mm,
        bottom_margin = 2Plots.mm,
        grid = false,
        framestyle = :box,
        tick_direction = :in,
        guidefontsize = 9,
        tickfontsize = 8,
        legendfontsize = 7,
        linewidth = 1.2,
        markerstrokewidth = 1.0,
        foreground_color_legend = nothing,
        background_color_legend = :white,  
        dpi = 600)


############################# HAMILTONIAN (Fig. 1) #############################

# Parameters used for the plots
# H_TFIM with J = 1, g = 0.5 on n = 3 qubits
# N = 1e7, δ = 0.05, drift ν = 0.001, 1000 trajectories

pauli_p01   = load("data/Fred_Hamiltonian_H_TFIM_p0.1_n3_N1e7_drift0.001.jld2")
pauli_p001  = load("data/Fred_Hamiltonian_H_TFIM_p0.01_n3_N1e7_drift0.001.jld2")
shadows_p01 = load("data/Shadows_Hamiltonian_H_TFIM_p0.1_n3_N1e7_drift0.001.jld2")

# ------------ (a) Energy ------------------------------------------------

residuals_pauli_p01   = pauli_p01["residuals"]
residuals_pauli_p001  = pauli_p001["residuals"]
residuals_shadows_p01 = shadows_p01["residuals"][:, 1]

# Certification radius averaged over the trajectories (dashed lines)
radius_pauli_p01   = mean(pauli_p01["real_deviation"])
radius_pauli_p001  = mean(pauli_p001["real_deviation"])
radius_shadows_p01 = mean(shadows_p01["radii"])

# Trajectories whose error is larger than their own certification radius
for (p, data) in [(0.1, pauli_p01), (0.01, pauli_p001)]
    outside = abs.(data["residuals"]) .> data["real_deviation"]
    println("Pauli IS, p = $p: $(sum(outside))/$(length(outside)) trajectories outside " *
            "the confidence region ($(round(100 * mean(outside), digits = 2))%)")
end

# Same bins for all the histograms
bins = range(-0.048767, 0.048767, length = 51)

plt = histogram(residuals_pauli_p01, bins = bins, normalize = :probability,
    fillcolor = :firebrick, fillalpha = 0.9, linewidth = 0.8,
    label = raw"Pauli IS, $p = 0.1$",
    xlabel = raw"$h_T - h_U$",
    ylabel = "Frequency (1000 trajectories)",
    ylims = (0, 0.27),
    legend = :topleft,
    left_margin = 4Plots.mm)                 

histogram!(plt, residuals_shadows_p01, bins = bins, normalize = :probability,
    fillcolor = :RoyalBlue, fillalpha = 0.8, linewidth = 0.8,
    label = raw"Shadows, $p = 0.1$")

histogram!(plt, residuals_pauli_p001, bins = bins, normalize = :probability,
    fillcolor = :salmon, fillalpha = 0.5, linewidth = 0.7,
    label = raw"Pauli IS, $p = 0.01$")

vline!(plt, [-radius_shadows_p01, radius_shadows_p01],
    color = :RoyalBlue, linewidth = 1.4, linestyle = :dash, label = false)
vline!(plt, [-radius_pauli_p01, radius_pauli_p01],
    color = :firebrick, linewidth = 1.4, linestyle = :dash, label = false)
vline!(plt, [-radius_pauli_p001, radius_pauli_p001],
    color = :salmon, linewidth = 1.4, linestyle = :dash, label = false)


# ------------ (b) Many observables ------------------------------------------------

# Observables in the shadow data:
columns = [2, 4, 5, 6, 7, 8, 9, 10]
labels  = [raw"$m_x$", raw"$m_z$", raw"$C_{xx}(1)$", raw"$C_{xx}(2)$",
           raw"$C_{yy}(1)$", raw"$C_{yy}(2)$", raw"$C_{zz}(1)$", raw"$C_{zz}(2)$"]

# Averages over the trajectories
estimates  = vec(mean(shadows_p01["estimates"][:, columns], dims = 1))
use_values = vec(mean(shadows_p01["true_use_averages"][:, columns], dims = 1))

x = 1:8
plt = scatter(x, estimates, yerror = fill(radius_shadows_p01, 8),
    markersize = 5, color = :RoyalBlue, markerstrokecolor = :RoyalBlue,
    label = false,
    xticks = (x, labels), xrotation = 45,
    xlims = (0.4, 8.6),
    ylims = (-0.13, 0.23),
    legend = (0.10, 0.90),        # top-left corner of the legend, in fractions of the axes
    legend_hfactor = 0.9)         # vertical spacing between the legend entries

scatter!(plt, x, use_values, marker = :diamond, markersize = 3.5,
    color = :gold, markerstrokecolor = :black, markerstrokewidth = 0.8,
    label = false)

hline!(plt, [0], color = :gray, linestyle = :dot, linewidth = 1.0, label = false)

# Legends
plot!(plt, [NaN], [NaN], linealpha = 0, marker = :circle,
    color = :RoyalBlue, markerstrokecolor = :RoyalBlue,
    label = "Shadow estimate")
plot!(plt, [NaN], [NaN], linealpha = 0, marker = :diamond,
    color = :gold, markerstrokecolor = :black, markerstrokewidth = 0.8,
    label = "Use-round value")

############################# WITNESS (Fig. 2) #############################

# Parameters used for the plots
# N = 5e5, δ = 0.05, drift ν = 0.001, 1000 runs per protocol
# Static protocols:  ϵ = 0.04
# Dynamic protocols: p = 0.008 (Pauli IS) and p = 0.12 (shadows)

wit_pauli   = load("data/witness_partialEntg_N5e5_drift0.001.jld2")    # static and dynamic
wit_static  = load("data/witness_static_shadows_partialEntg_N5e5_drift0.001_eps0.04.jld2")
wit_dynamic = load("data/witness_shadows_partialEntg_N5e5_drift0.001_p0.12.jld2")

ϵ_static = 0.04    # radius of the static protocols 

# The shadow protocols use the best of the 144 local-Clifford conjugates of W.
# Static: the witness is chosen on the verification block.
# Dynamic: in each run, the witness with the smallest estimate.
n_runs = size(wit_dynamic["est_shadows"], 1)
best   = [argmin(wit_dynamic["est_shadows"][i, :]) for i in 1:n_runs]
est_dynamic_shadows  = [wit_dynamic["est_shadows"][i, best[i]] for i in 1:n_runs]
true_dynamic_shadows = [wit_dynamic["true_shadows"][i, best[i]] for i in 1:n_runs]


protocols = [
    ("Static, Pauli IS",
        wit_pauli["true_W_verif"],
        wit_pauli["est_W_verif"] .+ ϵ_static,
        :salmon, 1.0),
    ("Static, Shadows",
        wit_static["true_best_static_shadows"],
        wit_static["est_best_static_shadows"] .+ ϵ_static,
        :lightsteelblue, 0.7),
    ("Dynamic, Pauli IS",
        wit_pauli["true_W_cert"],
        wit_pauli["est_W_cert"] .+ wit_pauli["epsilons_cert"],
        :firebrick, 1.0),
    ("Dynamic, Shadows",
        true_dynamic_shadows,
        est_dynamic_shadows .+ wit_dynamic["radii_shadows"],
        :RoyalBlue, 0.7)]

plt = plot(xlabel = raw"$w_U$",
    ylabel = raw"$w_{T} + \epsilon$",
    xlims = (-0.38, 0.68),
    ylims = (-0.38, 0.6),
    legend = :topleft,
    left_margin = 4Plots.mm)

# Grey region
plot!(plt, [0, 0.68, 0.68, 0], [-0.38, -0.38, 0, 0], seriestype = :shape,
    fillcolor = :grey, fillalpha = 0.08, linewidth = 0, label = false)
annotate!(plt, 0.35, -0.29, text("invalid certification", 7, :black, "Computer Modern"))
annotate!(plt, -0.33, -0.29, text("correct certification", 7, :black, :left, "Computer Modern"))

for (name, w_U, upper_bound, color, opacity) in protocols
    scatter!(plt, w_U, upper_bound, markersize = 1.8, color = color,
        markerstrokewidth = 0, markeralpha = opacity, label = false)

    # Legend entry 
    plot!(plt, [NaN], [NaN], linealpha = 0, marker = :circle, color = color,
        markerstrokewidth = 0, label = name)

    claimed = round(Int, 100 * mean(upper_bound .< 0))
    invalid = round(Int, 100 * mean((upper_bound .< 0) .& (w_U .>= 0)))
    println("$name: $claimed% claimed, $invalid% invalid")
end

hline!(plt, [0], color = :black, linestyle = :dot, linewidth = 0.8, label = false)
vline!(plt, [0], color = :black, linestyle = :dot, linewidth = 0.8, label = false)


############################# IBM EXPERIMENT (Fig. 3) #############################

# Parameters used for the plots
# 64 Bell pairs, p = 0.01, δ = 0.05

ibm = npzread("data/fidelity_p0.01.npz")

residuals = ibm["F_est"] .- ibm["F_exp"]
radius    = mean(ibm["epsilon"])    # test-round bound, averaged over the pairs (dashed lines)

outside = abs.(residuals) .> ibm["epsilon"]
println("IBM fidelity, p = 0.01: $(sum(outside))/$(length(outside)) trajectories outside " *
        "the confidence region ($(round(100 * mean(outside), digits = 2))%)")
println("Error for use rounds : ", mean(ibm["epsilon_exact"]))
println("Error for test rounds: ", radius)

# 30 bins, wide enough to show the dashed lines
edge = 1.1 * max(maximum(abs.(residuals)), radius)

plt = histogram(residuals, bins = range(-edge, edge, length = 31), normalize = :probability,
    fillcolor = :RoyalBlue, fillalpha = 0.9, linewidth = 0.8,
    label = false,
    xlabel = raw"$F_T - F_U$",
    ylabel = "Frequency (64 trajectories)",
    ylims = (0, 0.13),
    left_margin = 4Plots.mm)

vline!(plt, [-radius, radius], color = :RoyalBlue, linewidth = 1.4, linestyle = :dash, label = false)