param n;                  # Número total de nodos
param k;                  # Número de nodos a visitar (k ≤ n)
param c {1..n,1..n};      # Matriz de costos entre nodos

var x {i in 1..n, j in 1..n} binary;   # x[i,j] = 1 si se recorre el arco i→j
var y {i in 1..n} binary;              # y[i] = 1 si el nodo i es visitado
var u {i in 2..n} >= 0, <= k-1;        # Variables MTZ para subtour elimination

minimize Total_Cost:
    sum {i in 1..n, j in 1..n} c[i,j] * x[i,j];

subject to SelectK:
    sum {i in 1..n} y[i] = k;

subject to OneOut {i in 1..n}:
    sum {j in 1..n: j != i} x[i,j] = y[i];

subject to OneIn {j in 1..n}:
    sum {i in 1..n: i != j} x[i,j] = y[j];

subject to NoLoops {i in 1..n}:
    x[i,i] = 0;

subject to MTZ {i in 2..n, j in 2..n: i != j}:
    u[i] - u[j] + k * x[i,j] <= k - 1;
