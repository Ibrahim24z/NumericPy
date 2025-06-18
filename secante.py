def metodo_secante(f, x0, x1, tol=1e-6, max_iter=100):
    tabla = []
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if f1 == f0:
            return None, "División por cero."
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        tabla.append((i+1, x0, x1, f0, f1, x2))
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return tabla, None