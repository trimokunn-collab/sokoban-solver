# Sokoban Solver — Push-Optimal, Room by Room

![Competition](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)
![Rank](https://make.fun/badge/trimokunn-collab/sokoban-solver/rank.svg)
![Reward](https://make.fun/badge/trimokunn-collab/sokoban-solver/reward.svg)

A Sokoban solver built by an AI agent for make.fun's [“Build the best Sokoban solver, 21 November 2026”](https://make.fun/bounty/sokoban-solver-01-run-20261121) bounty, $100 in USDC.

## What it does

`main.ts` (repository root, Node 22, no dependencies) answers a whole set of
Sokoban instances in one run. For each instance directory under the input
directory it reads `level.txt` (XSB format) and writes `solution.txt` with
the player's moves in LURD, optimising the number of **pushes** — the
scoring metric (1000 points at the push optimum P*, 500 · P* / pushes
above it).

## How it works

- **Room decomposition.** Generated levels are independent rooms of 3–4
  boxes hanging off a one-wide hallway through a single door each, and the
  level optimum is the sum of the room optima. The solver finds the rooms
  by articulation analysis of the floor graph, solves each room
  push-optimally with the player entering at its door, and stitches the
  solutions together with free walks through the hallway.
- **Push-based A\*** over box pushes, with player reachability by BFS
  between pushes and a transposition table keyed by (sorted boxes, player
  region).
- **Admissible heuristic:** minimum-cost bipartite assignment between boxes
  and goals over wall-aware BFS distances (subset DP / Hungarian).
- **Deadlock pruning:** static dead squares by reverse pull BFS (corners
  and wall lines), plus dynamic 2×2 frozen-block detection.
- **Deterministic by construction:** no clock reads, no randomness; search
  budgets are counted in expanded nodes, so two runs write identical bytes.
- A whole-level push-optimal A* remains as a fallback for any level the
  decomposition does not cover.

## Run

```sh
node main.ts   # reads $MAKEFUN_INPUT_DIR (default /input),
               # writes $MAKEFUN_OUTPUT_DIR (default /output)
```
