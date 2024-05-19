import time
from collections import deque

def bfs_with_time(graph, start):    
    visited = set()  
    queue = deque([start])  
    visited_order = []  

    start_time = time.time()

    while queue:
        
        node = queue.popleft()
        if node not in visited:
          
            visited.add(node)
            visited_order.append(node)

         
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    end_time = time.time()
    time_taken = end_time - start_time
    
    return visited_order, time_taken

if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    start_node = 'A'
    visited_nodes, duration = bfs_with_time(graph, start_node)
    print("..................sequential..................................")
    print(f"Visited nodes in BFS order: {visited_nodes}")
    print(f"Time taken: {duration} seconds")
    abc =  duration
    print("sequential time",abc)

#..............................PRAM...............................................
import multiprocessing
from collections import deque
import time

def parallel_bfs_worker(node, graph, visited, dist, next_frontier):
    for neighbor in graph[node]:
        if not visited[neighbor]:
            visited[neighbor] = True
            dist[neighbor] = dist[node] + 1
            next_frontier.append(neighbor)

def parallel_bfs(graph, start):
    manager = multiprocessing.Manager()
    visited = manager.dict()
    dist = manager.dict()
    next_frontier = manager.list()

    for node in graph.keys():
        visited[node] = False
        dist[node] = float('inf')

    visited[start] = True
    dist[start] = 0

    curr_frontier = [start]

    while curr_frontier:
        processes = []
        for node in curr_frontier:
            p = multiprocessing.Process(target=parallel_bfs_worker, args=(node, graph, visited, dist, next_frontier))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        curr_frontier = list(next_frontier)
        next_frontier[:] = []

    return dist

if __name__ == '__main__':
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    start_node = 'A'
    
    start_time = time.time()
    
    dist = parallel_bfs(graph, start_node)
    
    end_time = time.time()
    
    time_taken = end_time - start_time
    print("................................PRAM....................................")
    
    print(f"Distances from node {start_node}: {dict(dist)}")
    print(f"Time taken for parallel BFS: {time_taken:.6f} seconds")
    pram = time_taken
    print("pram time:",pram)

#.......................................MPI>..............................................
from mpi4py import MPI
import numpy as np
import time

ROOT = 0

def bfs(graph, n, start):
    visited = np.zeros(n, dtype=int)
    dist = np.full(n, -1, dtype=int)

    queue = []
    visited[start] = 1
    queue.append(start)
    dist[start] = 0

    while queue:
        u = queue.pop(0)
        for v in range(n):
            if graph[u][v] and not visited[v]:
                visited[v] = 1
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    graph = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0]
    ])

    # Broadcast graph to all processes
    graph = comm.bcast(graph, root=ROOT)

    n = len(graph)  # Number of nodes
    start = ord('A') - ord('A')  # Convert 'A' to index 0

    # Determine the number of nodes per process
    nodes_per_process = n // size
    remainder = n % size
    start_node = rank * nodes_per_process
    end_node = start_node + nodes_per_process + (1 if rank == size - 1 else 0)

    # Scatter the portion of the graph to each process
    local_graph = np.zeros((end_node - start_node, n), dtype=int)
    comm.Scatter(graph[start_node:end_node], local_graph, root=ROOT)

    start_time = time.time()

    dist = bfs(local_graph, n, start)

    all_dist = None
    if rank == ROOT:
        all_dist = np.empty(n, dtype=int)
    comm.Gather(dist, all_dist, root=ROOT)

    end_time = time.time()

    print("........................................MPI...............................")

    if rank == ROOT:
        print("Distances from node", chr(start + ord('A')))
        for i in range(n):
            print("Node", chr(i + ord('A')), ":", all_dist[i])

    if rank == ROOT:
        print("Execution time:", end_time - start_time, "seconds")

    mpiTime = end_time - start_time
    print("mpi time:",mpiTime)


  #performance analysis.....................................................................

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<performance>>>>>>>>>>>>>>")

    if abc < pram and abc < mpiTime:
        print("Sequential BFS is the fastest.")
    elif pram < abc and pram < mpiTime:
        print("PRAM BFS is the fastest.")
    else:
        print("MPI BFS is the fastest.")