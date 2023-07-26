using Random
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using CellListMap

function initialize_simulation(npart, rng, boxl)
    positions = [-boxl .+ rand(rng, SVector{2}) .* (2.0 * boxl) for _ in 1:npart]
    angles = -π .+ rand(rng, npart) .* (2.0 * π)

    return (positions, angles)
end

function random_angle(rng)
    return -π .+ rand(rng, SVector{2}) .* (2.0 * π)
end

function neighbors(particles, position, cutoff, boxl)
    new_neighbor = []

    for (i, p) in enumerate(particles)
        dist = p .- position
        # Periodic boundary conditions
        dist = @. dist - boxl * ceil(dist / boxl)
        real_dist = norm(dist)
        if real_dist < cutoff
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
    rng = Random.Xoshiro(124)
    n_particles = 10_000
    density = 2.0
    # Assuming 2D simulation
    box_length = √(n_particles / density)
    @show box_length
    eta = 0.1
    cutoff = 2.5
    τ = 0.01
    time_steps = 1e4

    # Open up a file for saving the trajectory
    file = open("trajectory.xyz", "w")

    # Create the positions
    (position, angles) = initialize_simulation(n_particles, rng, box_length)

    for t in 1:time_steps
        # Write the headers for the frames in the trajectory file
        println(file, n_particles)
        println(file, "Frame $t")

        all_neighbors = neighborlist(
            position, cutoff; unitcell=[box_length, box_length], parallel=false
        )
        for idx in 1:n_particles
            # Obtain the indices that are neighbors to the current particle
            list_idx = map(x -> x[1] == idx, all_neighbors)
            neighbor_list = map(x -> x[2], all_neighbors[list_idx])
            if isempty(neighbor_list)
                continue
            end

            # Compute the average angle based on the neighbors
            avg_θ = compute_average(neighbor_list, angles)

            # Compute the noise vector based on the average angle
            noise = eta .* random_angle(rng)
            noise_vector = avg_θ .+ noise

            # Update the position according to the Vicsek model
            for (ijx, p) in enumerate(position)
                position[ijx] = @. p + (τ * noise_vector)
            end

            # Enforce periodic boundary conditions
            for (ijx, p) in enumerate(position)
                position[ijx] = @. p - box_length * ceil(p / box_length)
            end

            # Update the angles with the new vectors
            angles[idx] = atan(noise_vector...)
        end

        # Write every certain number of time steps
        if mod(t, 100) == 0
            current_directions = angle_to_vector.(angles)
            write_directions = reduce(hcat, current_directions)
            write_positions = reduce(hcat, position)
            writedlm(file, [write_positions' write_directions'])
        end
    end

    close(file)

    return nothing
end

main()
