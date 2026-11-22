// Sokoban solver — make.fun bounty "Build the best Sokoban solver, 21 November 2026"
// (competition sokoban-solver-01-run-20261121, harness program/v1).
//
// Contract (from the competition rules):
//   - The grader runs `node /bot/main.ts` once per set, Node 22, no network.
//   - A set holds 5 instances (i001..i005). Each instance is a Sokoban level in
//     XSB text form (# wall, space floor, . goal, $ box, * box on goal,
//     @ player, + player on goal), one row per line, in a file named level.txt
//     under the instance's directory in the input directory.
//   - For every instance the program writes solution.txt into the matching
//     directory of the output directory: the player's moves in LURD
//     (u/d/l/r, one letter per step, either case; whitespace ignored).
//     An empty file is a valid (scoreless) answer; an illegal step scores 0.
//   - Scoring rewards few pushes: 1000 points at the push optimum P*, and
//     500 * P* / pushes above it, so this solver optimises pushes, not moves.
//   - One run has 60 s of wall time for the whole set and two executions must
//     produce byte-identical output, so the search budgets below are counted
//     in expanded nodes, never in time. This program reads no clock, uses no
//     randomness, and iterates everything in a fixed order.
//
// Strategy:
//   The generator builds each level from independent rooms (1-4 rooms of 3-4
//   boxes) hanging off a one-wide hallway through a single door each; a box
//   can never leave its room in an optimal play, and the level optimum P* is
//   the sum of the room optima. So the solver first tries to decompose the
//   level into rooms (articulation analysis of the floor graph), solves each
//   room push-optimally with the player entering at its door, and stitches
//   the room solutions together with free walks through the hallway. If the
//   decomposition does not cover the level, it falls back to a whole-level
//   push-optimal A* over all boxes.
//
//   Both searches are push-based A* with:
//     - player reachability by BFS between pushes,
//     - an admissible heuristic: minimum-cost bipartite assignment between
//       boxes and goals over wall-aware BFS distances (each box needs at
//       least that many pushes), solved exactly by subset DP / Hungarian,
//     - static deadlock pruning: squares from which no box can ever reach a
//       goal (reverse pull BFS) are dead and never receive a box,
//     - a transposition table keyed by (sorted boxes, player region).

import {
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

const DIRS = [
  { dx: 0, dy: -1, walk: "u", push: "U" },
  { dx: 0, dy: 1, walk: "d", push: "D" },
  { dx: -1, dy: 0, walk: "l", push: "L" },
  { dx: 1, dy: 0, walk: "r", push: "R" },
];
const BIG = 1000000000;

// ---------------------------------------------------------------------------
// Level parsing (XSB)
// ---------------------------------------------------------------------------

function parseLevel(text) {
  const rows = text.replace(/\r\n?/g, "\n").split("\n");
  while (rows.length > 0 && rows[rows.length - 1] === "") rows.pop();
  while (rows.length > 0 && rows[0] === "") rows.shift();
  const H = rows.length;
  let W = 0;
  for (const r of rows) W = Math.max(W, r.length);
  const wall = new Uint8Array(W * H).fill(1);
  const goals = [];
  const goalSet = new Set();
  const boxes = [];
  let player = -1;
  for (let y = 0; y < H; y++) {
    const row = rows[y];
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * W + x;
      if (ch === "#") continue;
      wall[i] = 0;
      if (ch === "." || ch === "*" || ch === "+") {
        goals.push(i);
        goalSet.add(i);
      }
      if (ch === "$" || ch === "*") boxes.push(i);
      if (ch === "@" || ch === "+") player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { W, H, wall, goals, goalSet, boxes, player };
}

// ---------------------------------------------------------------------------
// Flood fill over cells not in `blocked` (Uint8Array). Returns BFS distances
// and parents, used both for player reachability and for walk reconstruction.
// ---------------------------------------------------------------------------

function flood(W, H, blocked, start) {
  const dist = new Int32Array(W * H).fill(-1);
  const prev = new Int32Array(W * H).fill(-1);
  if (start < 0 || blocked[start]) return { dist, prev };
  const queue = [start];
  dist[start] = 0;
  for (let head = 0; head < queue.length; head++) {
    const c = queue[head];
    const cx = c % W;
    const cy = (c / W) | 0;
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      const n = ny * W + nx;
      if (blocked[n] || dist[n] >= 0) continue;
      dist[n] = dist[c] + 1;
      prev[n] = c;
      queue.push(n);
    }
  }
  return { dist, prev };
}

// Walk moves (lowercase LURD) from `from` to `to` along flood parents.
function pathMoves(W, prev, from, to) {
  if (from === to) return "";
  const cells = [];
  let c = to;
  while (c !== from && c >= 0) {
    cells.push(c);
    c = prev[c];
  }
  if (c !== from) return null; // unreachable
  cells.reverse();
  let out = "";
  let cur = from;
  for (const n of cells) {
    const delta = n - cur;
    if (delta === 1) out += "r";
    else if (delta === -1) out += "l";
    else if (delta === W) out += "d";
    else out += "u";
    cur = n;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Static deadlock analysis: a square is "live" when a box placed there can be
// pushed to some goal (ignoring the other boxes). Reverse pull BFS from the
// goals. Every other floor square is dead: a box pushed there can never be
// solved. This covers corner deadlocks and wall-line deadlocks.
// ---------------------------------------------------------------------------

function computeLive(W, H, effWall, goals) {
  const live = new Uint8Array(W * H);
  const queue = [];
  for (const g of goals) {
    if (!live[g]) {
      live[g] = 1;
      queue.push(g);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head];
    const cx = cur % W;
    const cy = (cur / W) | 0;
    for (const d of DIRS) {
      // The box came from cur - d, pushed in direction d; the player stood
      // at cur - 2d. Both squares must be open floor.
      const bx = cx - d.dx;
      const by = cy - d.dy;
      const sx = cx - 2 * d.dx;
      const sy = cy - 2 * d.dy;
      if (bx < 0 || by < 0 || bx >= W || by >= H) continue;
      if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
      const b = by * W + bx;
      const s = sy * W + sx;
      if (effWall[b] || effWall[s] || live[b]) continue;
      live[b] = 1;
      queue.push(b);
    }
  }
  return live;
}

// BFS distance from every cell to each goal (ignoring boxes): a lower bound
// on the pushes one box needs to reach that goal.
function goalDistances(W, H, effWall, goals) {
  const out = [];
  for (const g of goals) {
    const blocked = effWall;
    const dist = new Int32Array(W * H).fill(-1);
    const queue = [g];
    dist[g] = 0;
    for (let head = 0; head < queue.length; head++) {
      const c = queue[head];
      const cx = c % W;
      const cy = (c / W) | 0;
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const n = ny * W + nx;
        if (blocked[n] || dist[n] >= 0) continue;
        dist[n] = dist[c] + 1;
        queue.push(n);
      }
    }
    out.push(dist);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Minimum-cost assignment between boxes and goals (admissible heuristic).
// Square cost matrix padded with zero-cost dummies; -1 distance = impossible.
// Exact subset DP for small sizes, Hungarian algorithm otherwise.
// ---------------------------------------------------------------------------

function popcount(x) {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

function hungarian(a, N) {
  const u = new Float64Array(N + 1);
  const v = new Float64Array(N + 1);
  const p = new Int32Array(N + 1);
  const way = new Int32Array(N + 1);
  for (let i = 1; i <= N; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(N + 1).fill(Infinity);
    const used = new Uint8Array(N + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;
      for (let j = 1; j <= N; j++) {
        if (used[j]) continue;
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= N; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= N; j++) {
    if (p[j] > 0) total += a[p[j] - 1][j - 1];
  }
  return total;
}

function matchCost(boxes, distG) {
  const n = boxes.length;
  const m = distG.length;
  if (n === 0) return 0;
  if (m === 0) return Infinity;
  const N = Math.max(n, m);
  const a = [];
  for (let i = 0; i < N; i++) {
    const row = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      if (i < n && j < m) {
        const d = distG[j][boxes[i]];
        row[j] = d < 0 ? BIG : d;
      } else {
        row[j] = 0;
      }
    }
    a.push(row);
  }
  let total;
  if (N <= 12) {
    const size = 1 << N;
    const dp = new Float64Array(size).fill(Infinity);
    dp[0] = 0;
    for (let mask = 0; mask < size; mask++) {
      const i = popcount(mask);
      if (i >= N) continue;
      const cur = dp[mask];
      if (!Number.isFinite(cur)) continue;
      for (let j = 0; j < N; j++) {
        if (mask & (1 << j)) continue;
        const nmask = mask | (1 << j);
        const val = cur + a[i][j];
        if (val < dp[nmask]) dp[nmask] = val;
      }
    }
    total = dp[size - 1];
  } else {
    total = hungarian(a, N);
  }
  return total >= BIG / 2 ? Infinity : total;
}

// Dynamic deadlock check: a 2x2 block whose four cells are all walls or
// boxes freezes every box in it. If any of those boxes is not on a goal, the
// state can never be solved. Only the four 2x2 squares touching the pushed
// box's destination can newly form, so only those are tested.
function formsFrozenBlock(W, H, effWall, boxSet, goalSet, dest) {
  const dx = dest % W;
  const dy = (dest / W) | 0;
  for (const oy of [-1, 0]) {
    for (const ox of [-1, 0]) {
      const x0 = dx + ox;
      const y0 = dy + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= W || y0 + 1 >= H) continue;
      let frozen = true;
      let badBox = false;
      for (const yy of [y0, y0 + 1]) {
        for (const xx of [x0, x0 + 1]) {
          const c = yy * W + xx;
          if (boxSet.has(c)) {
            if (!goalSet.has(c)) badBox = true;
          } else if (!effWall[c]) {
            frozen = false;
          }
        }
      }
      if (frozen && badBox) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Push-based A* over one (sub-)board.
//
// opts:
//   effWall      Uint8Array: 1 = blocked (walls plus, for a room, everything
//                outside the room's cells)
//   boxes        sorted starting box cells
//   goals        goal cells of this (sub-)board
//   goalSet      Set of goal cells
//   startPlayer  the player's starting cell
//   budget       maximum node expansions (deterministic cut-off)
//   entryCell    the door cell the player entered through, or -1
//   boxForbidden Set of cells a box may never occupy (unless a goal), or null
//   requireExit  goal states must leave the player able to reach entryCell
//   noBoxOnEntryAtGoal  goal states must not park a box on entryCell
//
// Returns { moves, pushes, expansions } or null when no solution is found
// within the budget.
// ---------------------------------------------------------------------------

function solvePush(board, opts) {
  const { W, H } = board;
  const effWall = opts.effWall;
  const goals = opts.goals;
  const goalSet = opts.goalSet;
  const entryCell = opts.entryCell ?? -1;
  const boxForbidden = opts.boxForbidden ?? null;
  const requireExit = opts.requireExit ?? false;
  const noBoxOnEntryAtGoal = opts.noBoxOnEntryAtGoal ?? false;
  const budget = opts.budget;

  const live = computeLive(W, H, effWall, goals);
  const distG = goalDistances(W, H, effWall, goals);

  const startBoxes = [...opts.boxes].sort((x, y) => x - y);
  for (const b of startBoxes) {
    if (!live[b]) return null; // a box starts on a dead square: unsolvable
  }
  const h0 = matchCost(startBoxes, distG);
  if (!Number.isFinite(h0)) return null;

  const blockedWith = (boxes) => {
    const blocked = new Uint8Array(effWall);
    for (const b of boxes) blocked[b] = 1;
    return blocked;
  };

  const regionRep = (fl) => {
    let rep = -1;
    for (let i = 0; i < fl.dist.length; i++) {
      if (fl.dist[i] >= 0) {
        rep = i;
        break;
      }
    }
    return rep;
  };

  const isGoal = (node) => {
    for (const b of node.boxes) if (!goalSet.has(b)) return false;
    if (noBoxOnEntryAtGoal && entryCell >= 0 && !goalSet.has(entryCell)) {
      if (node.boxes.includes(entryCell)) return false;
    }
    if (requireExit && entryCell >= 0) {
      if (node.flood.dist[entryCell] < 0) return false;
    }
    return true;
  };

  const nodes = [];
  const bestG = new Map();
  const heap = [];
  let seq = 0;

  const heapPush = (item) => {
    heap.push(item);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!less(heap[i], heap[parent])) break;
      const t = heap[i];
      heap[i] = heap[parent];
      heap[parent] = t;
      i = parent;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let smallest = i;
        if (l < heap.length && less(heap[l], heap[smallest])) smallest = l;
        if (r < heap.length && less(heap[r], heap[smallest])) smallest = r;
        if (smallest === i) break;
        const t = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = t;
        i = smallest;
      }
    }
    return top;
  };
  const less = (x, y) =>
    x.f < y.f || (x.f === y.f && (x.h < y.h || (x.h === y.h && x.seq < y.seq)));

  const startFlood = flood(W, H, blockedWith(startBoxes), opts.startPlayer);
  const startNode = {
    boxes: startBoxes,
    player: opts.startPlayer,
    g: 0,
    h: h0,
    flood: startFlood,
    parent: -1,
    boxFrom: -1,
    boxTo: -1,
  };
  nodes.push(startNode);
  bestG.set(startBoxes.join(",") + "|" + regionRep(startFlood), 0);
  heapPush({ f: h0, h: h0, seq: seq++, idx: 0 });

  let expansions = 0;
  let goalIdx = -1;

  while (heap.length > 0) {
    const { idx } = heapPop();
    const node = nodes[idx];
    const key =
      node.boxes.join(",") + "|" + regionRep(node.flood);
    if (node.g > (bestG.get(key) ?? Infinity)) continue;
    if (isGoal(node)) {
      goalIdx = idx;
      break;
    }
    if (expansions >= budget) return null;
    expansions++;

    const boxSet = new Set(node.boxes);
    const fl = node.flood;
    for (let bi = 0; bi < node.boxes.length; bi++) {
      const b = node.boxes[bi];
      const bx = b % W;
      const by = (b / W) | 0;
      for (const d of DIRS) {
        const standX = bx - d.dx;
        const standY = by - d.dy;
        const destX = bx + d.dx;
        const destY = by + d.dy;
        if (standX < 0 || standY < 0 || standX >= W || standY >= H) continue;
        if (destX < 0 || destY < 0 || destX >= W || destY >= H) continue;
        const stand = standY * W + standX;
        const dest = destY * W + destX;
        if (fl.dist[stand] < 0) continue; // player cannot reach the push side
        if (effWall[dest] || boxSet.has(dest)) continue;
        if (!live[dest]) continue; // dead square: the box could never finish
        if (boxForbidden && boxForbidden.has(dest) && !goalSet.has(dest)) {
          continue;
        }
        const nextBoxes = [...node.boxes];
        nextBoxes[bi] = dest;
        nextBoxes.sort((x, y) => x - y);
        const nextSet = new Set(nextBoxes);
        if (formsFrozenBlock(W, H, effWall, nextSet, goalSet, dest)) continue;
        const ng = node.g + 1;
        const childFlood = flood(W, H, blockedWith(nextBoxes), b);
        const childKey = nextBoxes.join(",") + "|" + regionRep(childFlood);
        if (ng >= (bestG.get(childKey) ?? Infinity)) continue;
        const nh = matchCost(nextBoxes, distG);
        if (!Number.isFinite(nh)) continue;
        bestG.set(childKey, ng);
        const child = {
          boxes: nextBoxes,
          player: b, // after the push the player stands where the box was
          g: ng,
          h: nh,
          flood: childFlood,
          parent: idx,
          boxFrom: b,
          boxTo: dest,
        };
        nodes.push(child);
        heapPush({ f: ng + nh, h: nh, seq: seq++, idx: nodes.length - 1 });
      }
    }
  }

  if (goalIdx < 0) return null;

  // Collect the pushes from the start to the goal node.
  const pushes = [];
  for (let i = goalIdx; i > 0; i = nodes[i].parent) {
    pushes.push({ from: nodes[i].boxFrom, to: nodes[i].boxTo });
  }
  pushes.reverse();

  // Replay the pushes, filling in the player's walks by BFS.
  let player = opts.startPlayer;
  const boxSet = new Set(startBoxes);
  let moves = "";
  for (const p of pushes) {
    const delta = p.to - p.from;
    const stand = p.from - delta;
    const fl = flood(W, H, blockedWith([...boxSet]), player);
    const walk = pathMoves(W, fl.prev, player, stand);
    if (walk === null) return null; // cannot happen for a found path
    moves += walk;
    if (delta === 1) moves += "R";
    else if (delta === -1) moves += "L";
    else if (delta === W) moves += "D";
    else moves += "U";
    boxSet.delete(p.from);
    boxSet.add(p.to);
    player = p.from;
  }
  return { moves, pushes: pushes.length, expansions };
}

// ---------------------------------------------------------------------------
// Replay a move string on the whole board; returns the resulting boxes and
// player cell, or null when a step is illegal. Used to stitch room solutions
// into the level state.
// ---------------------------------------------------------------------------

function simulate(board, boxes0, player0, moves) {
  const { W, H, wall } = board;
  const boxSet = new Set(boxes0);
  let player = player0;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    let d = null;
    if (c === "u") d = DIRS[0];
    else if (c === "d") d = DIRS[1];
    else if (c === "l") d = DIRS[2];
    else if (c === "r") d = DIRS[3];
    else return null; // not a move letter: the stitch cannot be trusted
    const px = player % W;
    const py = (player / W) | 0;
    const nx = px + d.dx;
    const ny = py + d.dy;
    if (nx < 0 || ny < 0 || nx >= W || ny >= H) return null;
    const next = ny * W + nx;
    if (wall[next]) return null;
    if (boxSet.has(next)) {
      const bx = nx + d.dx;
      const by = ny + d.dy;
      if (bx < 0 || by < 0 || bx >= W || by >= H) return null;
      const beyond = by * W + bx;
      if (wall[beyond] || boxSet.has(beyond)) return null;
      boxSet.delete(next);
      boxSet.add(beyond);
    }
    player = next;
  }
  return { boxes: [...boxSet].sort((a, b) => a - b), player };
}

// ---------------------------------------------------------------------------
// Room decomposition. Removing a room's door cell disconnects the room's
// interior from the rest of the floor graph. A candidate interior is a
// component (after removing one cell) that holds at least one box, exactly
// as many boxes as goals (counting a goal on the door cell itself), no
// player, and at most a room slot of cells. Candidates contained in a bigger
// candidate are dropped, then rooms are accepted greedily by size while
// their boxes stay disjoint. The decomposition is used only when the rooms
// together cover every box and every goal of the level.
// ---------------------------------------------------------------------------

function findRooms(board) {
  const { W, H, wall, goals, goalSet, boxes, player } = board;
  const floorCells = [];
  for (let i = 0; i < W * H; i++) if (!wall[i]) floorCells.push(i);
  const boxSet = new Set(boxes);

  const componentsExcluding = (excl) => {
    const comp = new Int32Array(W * H).fill(-1);
    const comps = [];
    for (const start of floorCells) {
      if (start === excl || comp[start] >= 0) continue;
      const cells = [];
      const queue = [start];
      comp[start] = comps.length;
      for (let head = 0; head < queue.length; head++) {
        const c = queue[head];
        cells.push(c);
        const cx = c % W;
        const cy = (c / W) | 0;
        for (const d of DIRS) {
          const nx = cx + d.dx;
          const ny = cy + d.dy;
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          const n = ny * W + nx;
          if (wall[n] || n === excl || comp[n] >= 0) continue;
          comp[n] = comps.length;
          queue.push(n);
        }
      }
      comps.push(cells);
    }
    return comps;
  };

  // key -> { cells:Set, doors:Set }
  const candidates = new Map();
  for (const door of floorCells) {
    for (const cells of componentsExcluding(door)) {
      if (cells.length > 42) continue;
      const cellSet = new Set(cells);
      if (cellSet.has(player)) continue;
      let nb = 0;
      let ng = 0;
      for (const c of cells) {
        if (boxSet.has(c)) nb++;
        if (goalSet.has(c)) ng++;
      }
      if (nb < 1 || nb > 6) continue;
      if (goalSet.has(door)) ng++;
      if (nb !== ng) continue;
      const key = [...cells].sort((a, b) => a - b).join(",");
      if (!candidates.has(key)) {
        candidates.set(key, { cells: cellSet, doors: new Set() });
      }
      candidates.get(key).doors.add(door);
    }
  }

  const candList = [...candidates.values()];
  // Drop candidates strictly contained in another candidate.
  const maximal = candList.filter(
    (c) =>
      !candList.some(
        (o) =>
          o !== c &&
          o.cells.size > c.cells.size &&
          [...c.cells].every((x) => o.cells.has(x)),
      ),
  );
  maximal.sort((a, b) => b.cells.size - a.cells.size);

  const usedBoxes = new Set();
  const rooms = [];
  for (const cand of maximal) {
    let overlaps = false;
    for (const c of cand.cells) {
      if (boxSet.has(c) && usedBoxes.has(c)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) continue;
    // The door must be reachable from the player's start without entering
    // the room, so the player can walk up to it through the hallway.
    const blocked = new Uint8Array(wall);
    for (const c of cand.cells) blocked[c] = 1;
    const reach = flood(W, H, blocked, player);
    let door = -1;
    for (const d of [...cand.doors].sort((a, b) => a - b)) {
      if (reach.dist[d] >= 0) {
        door = d;
        break;
      }
    }
    if (door < 0) continue;
    const roomBoxes = [];
    const roomGoals = [];
    for (const c of cand.cells) {
      if (boxSet.has(c)) roomBoxes.push(c);
      if (goalSet.has(c)) roomGoals.push(c);
    }
    if (goalSet.has(door)) roomGoals.push(door);
    for (const b of roomBoxes) usedBoxes.add(b);
    rooms.push({
      cells: cand.cells,
      door,
      boxes: roomBoxes.sort((a, b) => a - b),
      goals: roomGoals.sort((a, b) => a - b),
    });
  }

  if (rooms.length === 0) return null;
  if (usedBoxes.size !== boxes.length) return null;
  const coveredGoals = new Set();
  for (const r of rooms) for (const g of r.goals) coveredGoals.add(g);
  if (coveredGoals.size !== goals.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

// ---------------------------------------------------------------------------
// Solve one parsed level; returns the LURD move string ("" when unsolved).
// `state.spent` accumulates expansions across the set so the whole run stays
// inside a deterministic work budget.
// ---------------------------------------------------------------------------

const ROOM_BUDGET = 400000;
const LEVEL_BUDGET = 3000000;
const SET_BUDGET = 25000000;

function solveLevel(board, state) {
  if (board.player < 0 || board.boxes.length === 0) return "";
  const { W, H, wall, goalSet } = board;

  const rooms = findRooms(board);
  if (rooms) {
    // Solve every room on its own first; the solutions are independent of
    // the order in which they are later stitched together.
    const solved = [];
    let failed = false;
    for (const room of rooms) {
      const roomCells = new Set(room.cells);
      roomCells.add(room.door);
      const effWall = new Uint8Array(W * H).fill(1);
      for (const c of roomCells) if (!wall[c]) effWall[c] = 0;
      const roomGoalSet = new Set(room.goals);
      const base = {
        effWall,
        boxes: room.boxes,
        goals: room.goals,
        goalSet: roomGoalSet,
        startPlayer: room.door,
        entryCell: room.door,
      };
      let best = null;
      let exitless = false;
      // Strict mode: the box may never sit on the door cell, so the hallway
      // stays clear and the player can always leave the room.
      if (!roomGoalSet.has(room.door)) {
        const strict = solvePush(board, {
          ...base,
          budget: ROOM_BUDGET,
          boxForbidden: new Set([room.door]),
          requireExit: true,
        });
        state.spent += strict ? strict.expansions : ROOM_BUDGET;
        if (strict) best = strict;
      }
      // Relaxed mode: the box may pass over the door cell but may not be
      // parked on it in the final state, and the player must be able to
      // reach the door at the end.
      const lowerBound = matchCost(
        room.boxes,
        goalDistances(W, H, effWall, room.goals),
      );
      if (!best || best.pushes > lowerBound) {
        const relaxed = solvePush(board, {
          ...base,
          budget: ROOM_BUDGET,
          requireExit: true,
          noBoxOnEntryAtGoal: true,
        });
        state.spent += relaxed ? relaxed.expansions : ROOM_BUDGET;
        if (relaxed && (!best || relaxed.pushes < best.pushes)) best = relaxed;
      }
      // Last resort for this room: allow the player to end up unable to
      // leave; such a room is stitched last.
      if (!best) {
        const open = solvePush(board, {
          ...base,
          budget: ROOM_BUDGET,
          requireExit: false,
          noBoxOnEntryAtGoal: true,
        });
        state.spent += open ? open.expansions : ROOM_BUDGET;
        if (open) {
          best = open;
          exitless = true;
        }
      }
      if (!best) {
        failed = true;
        break;
      }
      solved.push({ room, moves: best.moves, exitless });
    }

    if (!failed) {
      // Rooms the player can leave come first; an exitless room goes last.
      solved.sort((a, b) => Number(a.exitless) - Number(b.exitless));
      let boxes = [...board.boxes];
      let player = board.player;
      let out = "";
      let ok = true;
      for (const s of solved) {
        const blocked = new Uint8Array(wall);
        for (const b of boxes) blocked[b] = 1;
        const fl = flood(W, H, blocked, player);
        const walk = pathMoves(W, fl.prev, player, s.room.door);
        if (walk === null) {
          ok = false;
          break;
        }
        out += walk;
        player = s.room.door;
        const after = simulate(board, boxes, player, s.moves);
        if (!after) {
          ok = false;
          break;
        }
        out += s.moves;
        boxes = after.boxes;
        player = after.player;
      }
      if (ok && boxes.every((b) => goalSet.has(b))) return out;
    }
  }

  // Fallback: one push-optimal A* over the whole level.
  if (state.spent < SET_BUDGET) {
    const whole = solvePush(board, {
      effWall: wall,
      boxes: board.boxes,
      goals: board.goals,
      goalSet,
      startPlayer: board.player,
      budget: LEVEL_BUDGET,
    });
    state.spent += whole ? whole.expansions : LEVEL_BUDGET;
    if (whole) return whole.moves;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Instance discovery and the main run. Every level.txt found under the input
// directory (at the root or inside an instance directory) is answered with a
// solution.txt in the mirrored position of the output directory.
// ---------------------------------------------------------------------------

function findLevels(inputDir) {
  const found = [];
  const walk = (dir, rel, depth) => {
    let entries;
    try {
      entries = readdirSync(dir);
    } catch {
      return;
    }
    entries.sort();
    for (const name of entries) {
      const full = join(dir, name);
      let st;
      try {
        st = statSync(full);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        if (depth < 3) walk(full, rel ? join(rel, name) : name, depth + 1);
      } else if (name === "level.txt") {
        found.push({ levelPath: full, relDir: rel });
      }
    }
  };
  walk(inputDir, "", 0);
  found.sort((a, b) => (a.relDir < b.relDir ? -1 : a.relDir > b.relDir ? 1 : 0));
  return found;
}

function main() {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  const state = { spent: 0 };
  for (const inst of findLevels(inputDir)) {
    let answer = "";
    try {
      const text = readFileSync(inst.levelPath, "utf8");
      const board = parseLevel(text);
      if (state.spent < SET_BUDGET) {
        answer = solveLevel(board, state);
      }
    } catch {
      answer = "";
    }
    const outDir = inst.relDir ? join(outputDir, inst.relDir) : outputDir;
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, "solution.txt"), answer, "utf8");
  }
}

main();
