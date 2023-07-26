using Random
using StaticArrays
using LinearAlgebra
using DelimitedFiles

function cell_list_neighbor_search_2D(
    particles, box_size::Tuple{Float64,Float64}, cutoff_radius::Float64
)
    n_particles = length(particles)
    n_cells_x = ceil(Int, box_size[1] / cutoff_radius)
    n_cells_y = ceil(Int, box_size[2] / cutoff_radius)

    # Create a dictionary to store particles in each cell
    cells_dict = Dict{Tuple{Int,Int},Vector{Int}}()

    # Function to get the cell indices for a given position
    get_cell_indices(x, y) = ((x - 1) ÷ cutoff_radius, (y - 1) ÷ cutoff_radius)

    # Populate cells_dict with particle indices in each cell
    for (index, particle) in enumerate(particles)
        cell_indices = get_cell_indices(particle[1], particle[2])
        cell_key = Tuple(cell_indices)
        if haskey(cells_dict, cell_key)
            push!(cells_dict[cell_key], index)
        else
            cells_dict[cell_key] = [index]
        end
    end

    # Function to get neighboring cells for a given cell
    function neighboring_cells(cell_indices)
        return [
            ((cell_indices[1] + dx) % n_cells_x, (cell_indices[2] + dy) % n_cells_y) for
            dx in -1:1, dy in -1:1
        ]
    end

    # Function to get neighboring particle indices for a given particle
    function get_neighbors(index)
        particle = particles[index]
        cell_indices = get_cell_indices(particle[1], particle[2])
        neighboring_cell_indices = neighboring_cells(cell_indices)
        neighbors = Set{Int}()
        for cell_idx in neighboring_cell_indices
            cell_key = Tuple(cell_idx)
            if haskey(cells_dict, cell_key)
                append!(neighbors, cells_dict[cell_key])
            end
        end
        return neighbors
    end

    # Perform neighbor search for each particle
    neighbor_list = Vector{Set{Int}}(undef, n_particles)
    for i in 1:n_particles
        neighbor_list[i] = get_neighbors(i)
    end

    return neighbor_list
end

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
    n_particles = 500
    density = 2.0
    # Assuming 2D simulation
    box_length = √(n_particles / density)
    @show box_length
    eta = 0.1
    cutoff = 2.5
    τ = 0.01
    time_steps = 1e3
    count = 0

    # Open up a file for saving the trajectory
    file = open("trajectory.xyz", "w")
    # Write the headers
    println(file, n_particles)
    println(file, "")

    # Create the positions
    (position, angles) = initialize_simulation(n_particles, rng, box_length)

    for t in 1:time_steps
        for idx in 1:n_particles
            # Obtain the indices that are neighbors to the current particle
            neighbor_list = neighbors(position, position[idx], cutoff, box_length)

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
            current_direction = angle_to_vector(angles[idx])

            if mod(t, 100) == 0
                # Write to file the positions and the velocities
                line_to_write = [position[idx]' current_direction']
                writedlm(file, line_to_write)
            end
        end

        # Write every certain number of time steps
        if mod(t, 100) == 0
            println(file, n_particles)
            println(file, "Frame $count")
        end
    end

    close(file)

    return nothing
end

main()
