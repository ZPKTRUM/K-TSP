# k-Traveling Salesman Problem (k-TSP)  
Modelos Exactos, Metaheurísticas y Solver LKH 3.0

## Descripción del Proyecto
Este proyecto aborda el k-Traveling Salesman Problem (k-TSP), una variante del TSP donde se deben visitar exactamente **k nodos** de un grafo (con `k ≤ n`), minimizando el costo total del recorrido.

Se implementaron y compararon cuatro enfoques:

- **Modelo exacto en AMPL** usando restricciones MTZ.
- **Heurística híbrida ACO + Simulated Annealing** implementada en Python.
- **Greedy Constructivo**, diseñado para obtener soluciones rápidas como baseline.
- **Solver LKH 3.0**, utilizado como referencia por su alta eficacia en TSP.

El proyecto incluye pruebas sobre instancias TSPLIB y un caso real aplicado a logística de vacunación en el sur de Chile.

---

## Enfoques Implementados

| Método | Lenguaje / Herramienta | Tipo | Ventajas |
|--------|-------------------------|------|----------|
| Modelo Exacto (AMPL) | AMPL + Gurobi/CPLEX | Exacto | Solución óptima garantizada |
| ACO + SA | Python | Heurístico aproximado | Buen balance costo/tiempo |
| LKH 3.0 | Ejecutable + `.par` | Heurístico avanzado | Alta calidad en grandes instancias |
| Greedy Constructivo | Python | Muy rápido | Útil como baseline, ejecución instantánea |

---

## Resultados Experimentales

### Comparación completa entre métodos (Costo, Tiempo y GAP del Greedy)

| Instancia | k | AMPL Costo | AMPL Tiempo (s) | ACO+SA Costo | ACO+SA Tiempo (s) | LKH Costo | LKH Tiempo (s) | Greedy Costo | GAP (%) | Greedy Tiempo (s) |
|-----------|---|-------------|------------------|----------------|--------------------|------------|------------------|----------------|----------|---------------------|
| att48     | 22 | 11446.25 | 0.33 | 12227.81 | 32   | 10628  | 2.31  | 13189.03 | +24.10 | 0.0000 |
| berlin52  | 6  | 847.35   | 0.41 | 7634.23  | 38   | 7542   | 4.58  | 8114.21  | +7.59  | 0.0020 |
| bier127   | 77 | 14628.69 | 0.67 | 125471.19 | 28   | 118282 | 46.63 | 11606.32 | -1.87 | 0.0161 |
| d198      | 10 | 4865.35  | 0.96 | 15778.13 | 64   | 15780  | 1137  |11867.49  | -24.79 | 0.0159 |
| eil101    | 25 | 99.40    | 1.71 | 639.78   | 45   | 629    | 23.05 | 663.48   | +5.48 | 0.0158 |

Interpretación:
- **LKH** obtiene la mejor calidad global.
- **AMPL** domina en instancias pequeñas con tiempos mínimos.
- **ACO + SA** obtiene buenos resultados, pero más lentos que Greedy.
- **Greedy** es extremadamente rápido, llegando a competir e incluso superar costos en varios casos (GAP negativo).


---

## Autores

- Sergio Villegas Osores
- Azul Alanya Chota
