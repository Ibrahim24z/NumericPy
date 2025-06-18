import numpy as np

# Método de Krylov para encontrar el polinomio característico de una matriz
# Este método genera una secuencia de vectores de Krylov y utiliza la matriz resultante
# para resolver un sistema lineal que nos da los coeficientes del polinomio característico.
## Parámetros:
# A: Matriz cuadrada de coeficientes.
# b: Vector de términos independientes (normalmente un vector de unos o el vector propio inicial).
#  ## Retorna:
# Una tupla con los coeficientes del polinomio característico y los vectores
# de Krylov generados.

def metodo_krylov(A, b):
    n = len(b)
    X = [b]
    
    for _ in range(n):
        X.append(A @ X[-1])

    H = np.column_stack(X[:-1])  # Matriz de Krylov
    y = X[-1]

    # Resolver H·c = y
    try:
        c = np.linalg.solve(H, y)
    except np.linalg.LinAlgError:
        c, *_ = np.linalg.lstsq(H, y, rcond=None)

    # Coeficientes del polinomio característico: p(λ) = λ^n - c₁λ^{n-1} - c₂λ^{n-2} - ... - cₙ
    coeficientes = np.append(1, -c[::-1])  # Invertimos para que sea λ³ - c2λ² - c1λ - c0

    return coeficientes, X  # Devuelve polinomio y vectores de Krylov
