from mpi4py import MPI
import itertools
import numpy as np

N = 10

def calculate_tour_cost(tour, cost_matrix):
    cost = 0
    for i in range(N - 1):
        cost += cost_matrix[tour[i]][tour[i + 1]]
    cost += cost_matrix[tour[-1]][tour[0]]  # Return to the starting city
    return cost

def generate_random_cost_matrix():
    return np.random.randint(1, 11, size=(N, N)) #I am generation the cost to be 1 - 10

def parallel_tsp(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    cost_matrix = generate_random_cost_matrix()
    best_tour = list(range(N))
    best_cost = float('inf')

    # Distribute workload among processes
    start = rank * (N // size)
    end = (rank + 1) * (N // size) if rank != size - 1 else N

    local_tour = best_tour[start:end]
    local_best_cost = float('inf')
    local_best_tour = local_tour

    # Explore permutations for the assigned subset
    for permuted_tour in itertools.permutations(local_tour):
        current_cost = calculate_tour_cost(permuted_tour, cost_matrix)
        if current_cost < local_best_cost:
            local_best_cost = current_cost
            local_best_tour = permuted_tour

    # Gather local best tours and costs to determine the global best
    all_best_tours = comm.gather(local_best_tour, root=0)
    all_best_costs = comm.gather(local_best_cost, root=0)

    if rank == 0:
        # Determine the global best tour and cost
        for i in range(size):
            if all_best_costs[i] < best_cost:
                best_cost = all_best_costs[i]
                best_tour = all_best_tours[i]

    # Broadcast the global best tour and cost to all processes
    best_tour = comm.bcast(best_tour, root=0)
    best_cost = comm.bcast(best_cost, root=0)

    if rank == 0:
        # Output the best tour and its cost
        print(f"Best Tour: {best_tour}")
        print(f"Best Cost: {best_cost}")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    parallel_tsp(comm)