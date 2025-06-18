
# Método de Bisección para encontrar raíces de una función
# Este método divide un intervalo [a, b] en dos mitades y busca la raíz
# de la función f(x) donde f(a) y f(b) tienen signos opuestos.
## Parámetros:
# f: Función a evaluar.
# a: Límite inferior del intervalo.
# b: Límite superior del intervalo.
# tol: Tolerancia para el error (por defecto 0.01).
# max_iter: Número máximo de iteraciones (por defecto 100).
## Retorna:
# Una lista de tuplas con el historial de iteraciones y un mensaje de error si lo hay.

def metodo_biseccion(f, a, b, tol=0.01, max_iter=100):
    tabla = []
    if f(a) * f(b) >= 0:
        return None, "No hay cambio de signo en el intervalo."
    for i in range(1, max_iter + 1):
        n = (a + b) / 2
        fa, fb, fn = f(a), f(b), f(n)
        error = abs(n - tabla[-1][3]) if i > 1 else None
        tabla.append((i, a, b, n, fa, fb, fn, error))
        if error is not None and error < tol:
            break
        if fa * fn < 0:
            b = n
        else:
            a = n
    return tabla, None