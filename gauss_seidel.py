import numpy as np

# Método de Gauss-Seidel para resolver sistemas lineales Ax = b
# Este método iterativo mejora la solución de un sistema lineal en cada iteración.
# Parámetros:
# A: Matriz de coeficientes del sistema.
# b: Vector de términos independientes.
# x0: Solución inicial (opcional, por defecto es un vector de ceros
# tol: Tolerancia para el error (por defecto 1e-6).
# max_iter: Número máximo de iteraciones (por defecto 100).
## Retorna:

def metodo_gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    tabla = []
    for it in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        error = np.linalg.norm(x - x_old, ord=np.inf)
        tabla.append((it + 1, *x, error))
        if error < tol:
            break
    return tabla