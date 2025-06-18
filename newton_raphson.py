
# Método de Newton-Raphson para encontrar raíces de una función.
# Este método utiliza la derivada de la función para encontrar sucesivas aproximaciones
# a la raíz de la función f(x).
# Parámetros:
# f: Función a evaluar.
# df: Derivada de la función f.
# x0: Valor inicial para la aproximación de la raíz.
# tol: Tolerancia para el error (por defecto 0.1).
# max_iter: Número máximo de iteraciones (por defecto 100).
## Retorna:
# Una lista de tuplas con el historial de iteraciones y un mensaje de error si lo hay.




def metodo_newton(f, df, x0, tol=0.1, max_iter=100):
    tabla = []
    for i in range(1, max_iter + 1):
        fx, dfx = f(x0), df(x0)
        if dfx == 0:
            return None, "Derivada cero. Método no puede continuar."
        x1 = x0 - fx / dfx
        error = abs(x1 - x0) if i > 1 else None
        tabla.append((i, x0, fx, dfx, x1, error))
        if error is not None and error < tol:
            break
        x0 = x1
    return tabla, None