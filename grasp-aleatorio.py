import sys
import csv
import heapq
import random
from collections import defaultdict
import re
import os
import matplotlib.pyplot as plt
import networkx as nx

INF = 1e18

def read_graph(grafo_path):
    adj = defaultdict(list)
    nodes = set()
    with open(grafo_path, newline='') as f:
        rd = csv.reader(f)
        for row in rd:
            u = int(row[0])
            v = int(row[1])
            c = float(row[2])
            adj[u].append((v, c))
            nodes.add(u)
            nodes.add(v)
    return adj, nodes

def read_instance(inst_path):
    workers = []
    with open(inst_path, newline='') as f:
        rd = csv.reader(f)
        for row in rd:
            vi = int(row[0])
            ri = float(row[1])
            workers.append((vi, ri))
    return workers

def dijkstra(adj, source, nodes):
    dist = {u: INF for u in nodes}
    parent = {u: None for u in nodes}
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, c in adj.get(u, []):
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

def all_pairs_shortest_paths(adj, nodes):
    dist = {}
    parent = {}
    for u in nodes:
        dist[u], parent[u] = dijkstra(adj, u, nodes)
    return dist, parent

def reconstruct_path(parent, u, v):
    if u == v:
        return [u]
    if parent[u][v] is None:
        return []
    path = []
    cur = v
    while cur is not None and cur != u:
        path.append(cur)
        cur = parent[u][cur]
    if cur is None:
        return []
    path.append(u)
    return path[::-1]

def build_cover_candidates(workers, nodes, dist):
    cover = {v: set() for v in nodes}
    for w_idx, (vi, ri) in enumerate(workers):
        if vi not in dist:
            continue
        dist_from_vi = dist[vi]
        for v in nodes:
            d = dist_from_vi.get(v, INF)
            if d <= ri:
                cover[v].add(w_idx)
    return cover

def greedy_randomized_construct(adj, nodes, workers, dist, cover, rng, k_rcl=3):
    start = 0
    current = start
    uncovered = set(range(len(workers)))
    route = [start]
    if start in cover:
        uncovered -= cover[start]
    partials = []
    partials.append((route[:], len(uncovered)))
    while uncovered:
        scored = []
        for v in nodes:
            new_covered = cover.get(v, set()) & uncovered
            if not new_covered:
                continue
            c = dist.get(current, {}).get(v, INF)
            if c == INF:
                continue
            benefit = len(new_covered)
            score = benefit / c
            scored.append((score, v, benefit, c))
        if not scored:
            break
        scored.sort(reverse=True, key=lambda x: x[0])
        rcl = scored[:max(1, min(k_rcl, len(scored)))]
        chosen = rng.choice(rcl)
        chosen_node = chosen[1]
        route.append(chosen_node)
        uncovered -= cover.get(chosen_node, set())
        current = chosen_node
        partials.append((route[:], len(uncovered)))
    if route[-1] != start:
        route.append(start)
    total_cost = route_cost(route, dist)
    return route, total_cost, partials

def route_cost(route, dist):
    cost = 0.0
    for i in range(len(route)-1):
        u = route[i]
        v = route[i+1]
        d = dist.get(u, {}).get(v, INF)
        if d == INF:
            return INF
        cost += d
    return cost

def covers_all(route, cover, num_workers):
    served = set()
    for v in route:
        served |= cover.get(v, set())
    return len(served) == num_workers

def two_opt_random(route, dist, max_iterations=600, rng=None):
    if len(route) <= 3:
        return route, route_cost(route, dist)
    if rng is None:
        rng = random.Random()
    best = route[:]
    best_cost = route_cost(best, dist)
    for iteration in range(max_iterations):
        improved = False
        n = len(best)
        possible_moves = []
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                possible_moves.append((i, j))
        rng.shuffle(possible_moves)
        for i, j in possible_moves:
            candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
            c = route_cost(candidate, dist)
            if c < best_cost - 1e-9:
                best = candidate
                best_cost = c
                improved = True
                break
        if not improved:
            break
    return best, best_cost

def compute_max_p(best_cost, partials, dist, cover, num_workers):
    max_p = 0.0
    for pr, unc in partials:
        if unc == 0 or unc == num_workers:
            continue
        proute = pr[:]
        if proute[-1] != 0:
            proute.append(0)
        proute, pcost = two_opt_random(proute, dist)
        if pcost >= INF:
            continue
        ratio = (best_cost - pcost) / unc
        if ratio > max_p:
            max_p = ratio
    return max_p

def grasp_main_random(adj, nodes, workers, dist, cover, iters=100, k_rcl_start=5, seed=42, two_opt_iterations=600):
    rng = random.Random(seed)
    best_route = None
    best_cost = INF
    best_partials = None
    progress = []
    for it in range(1, iters+1):
        k_rcl = max(1, k_rcl_start - (it // (iters // 5)))
        route, cost, partials = greedy_randomized_construct(adj, nodes, workers, dist, cover, k_rcl=k_rcl, rng=rng)
        r1, c1 = two_opt_random(route, dist, max_iterations=two_opt_iterations, rng=rng)
        if r1[0] != 0:
            r1.insert(0, 0)
        if r1[-1] != 0:
            r1.append(0)
        if not covers_all(r1, cover, len(workers)):
            c1 = INF
        if c1 < best_cost:
            best_cost = c1
            best_route = r1
            best_partials = partials
            print(f"[Iter {it}] Nuevo mejor: costo={best_cost:.4f}, ruta_len={len(best_route)}")
        progress.append(best_cost if best_cost < INF else None)
    return best_route, best_cost, progress, best_partials

def expand_route(route, parent):
    expanded = []
    for i in range(len(route)-1):
        u = route[i]
        v = route[i+1]
        segment = reconstruct_path(parent, u, v)
        if not segment:
            return []
        if i > 0:
            segment = segment[1:]
        expanded.extend(segment)
    return expanded

def save_results(expanded_route, instance_path):
    idx = re.search(r'\d+', instance_path).group()
    outname = f"solucion{idx}.txt"
    with open(outname, 'w') as f:
        f.write(' '.join(map(str, expanded_route)))
    print(f"Ruta guardada en {outname}")
    return outname

def plot_progress(progress, out='plot_progress.png'):
    xs = list(range(1, len(progress)+1))
    ys = [p for p in progress]
    xs_f = xs[:len(ys)]
    plt.figure(figsize=(8,4))
    plt.plot(xs_f, ys, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor costo')
    plt.title('Progreso de la heurística (2-opt aleatorio)')
    plt.grid(True)
    plt.savefig(out)
    plt.close()
    print(f"Plot en {out}")

def plot_route(adj, expanded_route, nodes, workers, out='plot_route.png'):
    G = nx.Graph()
    for u in adj:
        for v, c in adj[u]:
            G.add_edge(u, v, weight=c)
    pos = nx.spring_layout(G, seed=42)
    workers_idx = [node[0] for node in workers]
    node_colors = []
    for node in G.nodes():
        if node == 0:
            node_colors.append('green')
        elif node in workers_idx:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')
    nx.draw(G, pos, node_size=10, font_size=8, edge_color='lightgray', node_color=node_colors)
    route_edges = [(expanded_route[i], expanded_route[i+1]) for i in range(len(expanded_route)-1)]
    nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='r', width=2)
    plt.title('Mejor ruta encontrada (2-opt aleatorio)')
    plt.savefig(out)
    plt.close()
    print(f"Plot de ruta en {out}")

def main():
    grafo_path = sys.argv[1]
    instance_path = sys.argv[2]
    print("Leyendo grafo...")
    adj, nodes = read_graph(grafo_path)
    print(f"Nodos: {len(nodes)}")
    print("Calculando distancias...")
    dist, parent = all_pairs_shortest_paths(adj, nodes)
    print("Distancias OK.")
    print(f"Procesando {instance_path} ...")
    print("Leyendo instancia...")
    workers = read_instance(instance_path)
    print(f"Trabajadores: {len(workers)}")
    print("Construyendo cobertura...")
    cover = build_cover_candidates(workers, nodes, dist)
    print(f"Nodos con cobertura: {sum(1 for v in cover if cover[v])}")
    ITERS = 1000
    K_RCL_START = 5
    SEED = 42
    TWO_OPT_ITER = 600
    print(f"GRASP (2-opt aleatorio): iters={ITERS}, k_rcl_start={K_RCL_START}, seed={SEED}")
    best_route, best_cost, progress, partials = grasp_main_random(
        adj, nodes, workers, dist, cover,
        iters=ITERS, k_rcl_start=K_RCL_START,
        seed=SEED,
        two_opt_iterations=TWO_OPT_ITER
    )
    print("Mejor ruta:")
    expanded_best_route = expand_route(best_route, parent)
    print(expanded_best_route)
    print(f"Costo: {best_cost:.4f}")
    max_p = compute_max_p(best_cost, partials, dist, cover, len(workers))
    print(f"Valor máximo de P: {max_p:.4f}")
    save_results(expanded_best_route, instance_path)
    plot_progress(progress, out=f"plot_progress_{os.path.basename(instance_path)}.png")
    plot_route(adj, expanded_best_route, nodes, workers, out=f"plot_route_{os.path.basename(instance_path)}.png")

if __name__ == "__main__":
    main()