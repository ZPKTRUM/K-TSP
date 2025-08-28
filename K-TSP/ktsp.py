import tsplib95
import numpy as np
import random
from sklearn.cluster import KMeans
import csv
from tqdm import tqdm
import math
import time

# --- Cargar problema TSPLIB ---
def load_problem(filename):
    problem = tsplib95.load(filename)
    nodes = list(problem.get_nodes())
    coords = np.array([problem.node_coords[i] for i in nodes])
    return problem, nodes, coords

# --- Matriz de distancias euclidianas ---
def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def create_distance_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = euclidean(coords[i], coords[j])
    return matrix

# --- Clustering con KMeans ---
def cluster_nodes(coords, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(coords)
    clusters = [[] for _ in range(k)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return clusters

# --- Costo de un tour ---
def calculate_tour_cost(tour, dist_matrix):
    return sum(dist_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))

# --- Simulated Annealing ---
def simulated_annealing(tour, dist_matrix, initial_temp=1000, final_temp=1, alpha=0.995, max_iter=500):
    current_tour = tour[:]
    current_cost = calculate_tour_cost(current_tour, dist_matrix)
    best_tour = current_tour[:]
    best_cost = current_cost
    temp = initial_temp

    for _ in range(max_iter):
        i, j = sorted(random.sample(range(1, len(tour)-1), 2))
        new_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
        new_cost = calculate_tour_cost(new_tour, dist_matrix)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_tour = new_tour
            current_cost = new_cost
            if new_cost < best_cost:
                best_tour = new_tour
                best_cost = new_cost

        temp *= alpha
        if temp < final_temp:
            break

    return best_tour

# --- Clase Hormiga ---
class Ant:
    def __init__(self, start, unvisited, dist_matrix, alpha=1, beta=2):
        self.start = start
        self.tour = [start]
        self.unvisited = set(unvisited)
        self.dist_matrix = dist_matrix
        self.alpha = alpha
        self.beta = beta

    def select_next(self, pheromones):
        current = self.tour[-1]
        probs = []

        for city in self.unvisited:
            dist = self.dist_matrix[current][city]
            if dist == 0:
                continue
            tau = pheromones[current][city] ** self.alpha
            eta = (1 / dist) ** self.beta
            probs.append((city, tau * eta))

        if not probs:
            return random.choice(list(self.unvisited))

        total = sum(p for _, p in probs)
        probs = [(city, p / total) for city, p in probs]

        r = random.random()
        cumulative = 0.0
        for city, prob in probs:
            cumulative += prob
            if r <= cumulative:
                return city
        return probs[-1][0]

    def build_tour(self, pheromones):
        while self.unvisited:
            next_city = self.select_next(pheromones)
            self.tour.append(next_city)
            self.unvisited.remove(next_city)
        self.tour.append(self.start)

# --- Algoritmo k-TSP con ACO + SA ---
def k_tsp_aco(distance_matrix, coords, k=3, iterations=100, ants_per_iter=10):
    n = len(distance_matrix)
    pheromones = np.ones((n, n))
    best_total = float('inf')
    best_solution = []

    clusters = cluster_nodes(coords, k)

    for _ in range(iterations):
        all_tours = []
        all_dists = 0

        for cluster in clusters:
            if len(cluster) <= 2:
                continue

            start = cluster[0]
            best_cluster_tour = None
            best_cluster_dist = float('inf')

            for _ in range(ants_per_iter):
                ant = Ant(start, cluster[1:], distance_matrix)
                ant.build_tour(pheromones)

                refined_tour = simulated_annealing(ant.tour, distance_matrix)
                refined_dist = calculate_tour_cost(refined_tour, distance_matrix)

                if refined_dist < best_cluster_dist:
                    best_cluster_dist = refined_dist
                    best_cluster_tour = refined_tour

            all_tours.append(best_cluster_tour)
            all_dists += best_cluster_dist

            for i in range(len(best_cluster_tour) - 1):
                a, b = best_cluster_tour[i], best_cluster_tour[i+1]
                pheromones[a][b] += 1.0 / best_cluster_dist
                pheromones[b][a] += 1.0 / best_cluster_dist

        if all_dists < best_total:
            best_total = all_dists
            best_solution = all_tours

        pheromones *= 0.95

    return best_solution, best_total

# --- Algoritmo k-TSP Greedy ---
def k_tsp_greedy(distance_matrix, coords, k=3):
    clusters = cluster_nodes(coords, k)
    total_cost = 0
    all_tours = []

    for cluster in clusters:
        if len(cluster) <= 2:
            continue

        unvisited = set(cluster)
        start = cluster[0]
        tour = [start]
        unvisited.remove(start)

        current = start
        while unvisited:
            next_city = min(unvisited, key=lambda x: distance_matrix[current][x])
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        tour.append(start)
        cost = calculate_tour_cost(tour, distance_matrix)
        total_cost += cost
        all_tours.append(tour)

    return all_tours, total_cost

# --- Ejecutar todas las instancias ---
def run_all_instances():
    bks_dict = {
        "rl1323.tsp": 270199,  # Best Known Solution (BKS) 
        # Puedes agregar más instancias aquí
    }

    k_values = [270]  # Puedes ajustar valores de k
    results = []

    print("\nIniciando procesamiento de instancias...\n")

    for file in tqdm(bks_dict, desc="Archivos", position=0):
        try:
            problem, nodes, coords = load_problem(file)
            distance_matrix = create_distance_matrix(coords)

            bks = bks_dict[file]
            lkh = bks

            iters = 30 if len(coords) > 1000 else 50
            ants = 3 if len(coords) > 1000 else 5

            for k in tqdm(k_values, desc=f"     {file}", position=1, leave=False):
                if k > len(coords):
                    continue

                # ACO + SA
                start_aco = time.time()
                _, cost_aco = k_tsp_aco(distance_matrix, coords, k=k, iterations=iters, ants_per_iter=ants)
                end_aco = time.time()
                time_aco = round(end_aco - start_aco, 4)
                gap_aco = ((cost_aco - lkh) / lkh) * 100 if lkh else None
                results.append([file, bks, lkh, k, round(cost_aco, 2), round(gap_aco, 2), time_aco, "ACO+SA"])

                # Greedy
                start_greedy = time.time()
                _, cost_greedy = k_tsp_greedy(distance_matrix, coords, k=k)
                end_greedy = time.time()
                time_greedy = round(end_greedy - start_greedy, 4)
                gap_greedy = ((cost_greedy - lkh) / lkh) * 100 if lkh else None
                results.append([file, bks, lkh, k, round(cost_greedy, 2), round(gap_greedy, 2), time_greedy, "Greedy"])

        except Exception as e:
            tqdm.write(f"Error en {file}: {e}")

    # Guardar resultados
    with open("resultados_k_tsp.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo", "BKS", "Costo LKH", "k", "Costo", "Gap (%)", "Tiempo (s)", "Algoritmo"])
        writer.writerows(results)

    print("\nResultados guardados en 'resultados_k_tsp.csv'")

if __name__ == '__main__':
    run_all_instances()
