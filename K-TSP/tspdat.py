import tsplib95
import numpy as np

def tsp_to_ampl_dat(tsp_file, k, output_file):
    problem = tsplib95.load(tsp_file)
    nodes = list(problem.get_nodes())
    n = len(nodes)
    coords = [problem.node_coords[i] for i in nodes]

    def euclidean(p1, p2):
        return round(np.linalg.norm(np.array(p1) - np.array(p2)), 2)

    matrix = [[euclidean(coords[i], coords[j]) for j in range(n)] for i in range(n)]

    with open(output_file, 'w') as f:
        f.write(f"param n := {n};\n")
        f.write(f"param k := {k};\n\n")
        f.write("param c :\n")
        f.write("   " + "  ".join(str(j+1) for j in range(n)) + " :=\n")

        for i in range(n):
            row = " ".join(f"{matrix[i][j]}" for j in range(n))
            f.write(f"{i+1} {row}\n")

        f.write(";\n")

# Ejemplo de uso
tsp_to_ampl_dat("eil101.tsp", k=15, output_file="eil101.dat")