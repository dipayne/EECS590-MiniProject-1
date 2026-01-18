
# Mini Project 1
# 01/16/26
# Author: Davis Payne

import math

class MarkovRewardProcess:
    def __init__(self, states, rewards, transitions, gamma=0.9):
        self.states = states
        self.R = rewards          # dict: R[s]
        self.P = transitions      # dict: P[s][s']
        self.gamma = gamma

    def value_iteration(self, tol=1e-6, max_iter=1000):
        V = {s: 0.0 for s in self.states}

        for iteration in range(max_iter):
            delta = 0.0
            V_new = V.copy()

            for s in self.states:
                V_new[s] = self.R[s] + self.gamma * sum(
                    self.P[s][s_next] * V[s_next] for s_next in self.P[s]
                )
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            if delta < tol:
                print(f"Converged in {iteration} iterations.")
                break

        return V


class GridWorldMRP(MarkovRewardProcess):
    def __init__(self, mask, gamma=0.9):
        self.rows = len(mask)
        self.cols = len(mask[0])
        self.state_map = {}
        self.states = []
        idx = 0

        for i in range(self.rows):
            for j in range(self.cols):
                if mask[i][j] != -math.inf:
                    self.state_map[(i, j)] = idx
                    self.states.append(idx)
                    idx += 1

        rewards = {self.state_map[(i, j)]: mask[i][j]
                   for (i, j) in self.state_map}

        transitions = {s: {} for s in self.states}

        for (i, j), s in self.state_map.items():
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (ni, nj) in self.state_map:
                    neighbors.append(self.state_map[(ni, nj)])

            if not neighbors:
                transitions[s][s] = 1.0
            else:
                p = 1.0 / len(neighbors)
                for ns in neighbors:
                    transitions[s][ns] = p

        super().__init__(self.states, rewards, transitions, gamma)


def main():
    mask = [
        [0, -math.inf, -1, -1, -1],
        [-1, -math.inf, -1, -math.inf, -1],
        [-1, -math.inf, -1, -math.inf, -1],
        [-1, -1, -1, -math.inf, -1],
        [-3, -3, -3, -3, -3]
    ]

    grid = GridWorldMRP(mask, gamma=0.9)
    V = grid.value_iteration()

    print("\nState Values:")
    for (i, j), s in grid.state_map.items():
        print(f"Cell {(i, j)} -> Value: {V[s]:.3f}")


if __name__ == "__main__":
    main()
