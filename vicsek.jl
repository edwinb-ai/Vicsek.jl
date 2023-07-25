using Random
using StaticArrays
using LinearAlgebra

function initialize_simulation(npart, rng, boxl)
    positions = [-boxl .+ rand(rng, SVector{2}) .* (2.0 * boxl) for _ in 1:npart]
    angles = -π .+ rand(rng, npart) .* (2.0 * π)

    return (positions, angles)
end

function neighbors(particles, position, cutoff)
    new_neighbor = []

    for (i, p) in enumerate(particles)
        dist = norm(p .- position)
        if dist < cutoff
            new_neighbor = append!(new_neighbor, i)
        end
    end

    return new_neighbor
end

function unit_vector(v1, v2)
    vector = v1 .- v2
    dist = norm(vector)

    return vector ./ dist
end

function angle_to_vector(θ)
    x = cos(θ)
    y = sin(θ)

    v1 = @SVector [x, y]
    v2 = @SVector zeros(2)

    return unit_vector(v1, v2)
end

function compute_average(neighs, angles)
    avg_vector = @SVector zeros(2)

    for idx in neighs
        θ_vec = angle_to_vector(angles[idx])
        avg_vector = avg_vector .+ θ_vec
    end

    return avg_vector ./ length(neighs)
end

function main()
    rng = Random.Xoshiro(123)
    n_particles = 10
    density = 1.0
    # Assuming 2D simulation
    box_length = √(n_particles / density)
    eta = 0.1
    cutoff = 2.0
    τ = 0.01

    # Create the positions
    (position, angles) = initialize_simulation(n_particles, rng, box_length)

    neighbor_list = neighbors(position, position[1], cutoff)
    avg_θ = compute_average(neighbor_list, angles)
    @show avg_θ

    return nothing
end

main()
