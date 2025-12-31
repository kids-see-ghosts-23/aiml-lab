import numpy as np
from queue import PriorityQueue

def heuristic(state, goal):
    return np.count_nonzero(state != goal)

def get_moves(state):
    moves = []
    x, y = np.where(state == 0)
    #x, y = x[0], y[0]
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new = state.copy()
            new[x, y], new[nx, ny] = new[nx, ny], new[x, y]
            moves.append(new)
    return moves

def solve(initial, goal):
    pq = PriorityQueue()
    count = 0  # <--- Unique counter to prevent comparison errors
    # Store: (priority, counter, current_state, parent_asstring)
    pq.put((0, count, initial, None))
    visited = {}
    while not pq.empty():
        _ , _ , state, parent = pq.get()
        key = str(state)
        if key in visited:
            continue
        visited[key] = parent
        if np.array_equal(state, goal):
            # Reconstruct path using the strings stored in visited
            path = []
            curr_str = key
            while curr_str is not None:
                path.append(curr_str)
                curr_str = visited[curr_str]
            return path[::-1]
        for nxt in get_moves(state):
            if str(nxt) not in visited:
                count += 1
                priority = heuristic(nxt, goal)
                pq.put((priority, count, nxt, key))
    return None

if __name__ == "__main__":
    start = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])
    goal = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
    solution = solve(start, goal)
    if solution:
        for step in solution:
            print(step)
            print("---")
        print(f"Solved in {len(solution)-1} moves!")