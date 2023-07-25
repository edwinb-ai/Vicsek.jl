using Random
using StaticArrays
using LinearAlgebra

function initialize_simulation(npart, rng, boxl)
    positions = [-boxl .+ rand(rng, SVector{2}) .* (2.0 * boxl) for _ in 1:npart]
    angles = -π .+ rand(rng, npart) .* (2.0 * π)

    return (positions, angles)
end

function random_angle(rng)
    return -π .+ rand(rng, SVector{2}) .* (2.0 * π)
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
    v1 = @SVector [cos(θ), sin(θ)]
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
    @show box_length
    eta = 0.45
    cutoff = 2.0
    τ = 0.01
    init_time = 0.0
    final_time = 10.0

    # Create the positions
    (position, angles) = initialize_simulation(n_particles, rng, box_length)
    display(position)

    while init_time < final_time
        for idx in 1:n_particles
            neighbor_list = neighbors(position, position[idx], cutoff)
            avg_θ = compute_average(neighbor_list, angles)
            noise = eta .* random_angle(rng)
            noise_vector = avg_θ .+ noise
            for (idx, p) in enumerate(position)
                position[idx] = @. p + τ * noise_vector
            end
            angles[idx] = atan(noise_vector[1], noise_vector[2])
        end

        init_time += τ
    end

    display(position)


    return nothing
end

main()
