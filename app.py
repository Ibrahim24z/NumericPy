
# Importar bibliotecas necesarias
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sympy import symbols, sympify, lambdify, diff
import matplotlib.pyplot as plt
import numpy as np
import csv
from tkinter import filedialog


#Importar metodos numéricos
from metodos.biseccion import metodo_biseccion
from metodos.gauss_seidel import metodo_gauss_seidel
from metodos.krylov import metodo_krylov
from metodos.newton_raphson import metodo_newton
from metodos.secante import metodo_secante


# Variables globales para almacenar resultados y encabezados
# Esto permite exportar los resultados después de ejecutar un método
ultimo_resultado = []
ultimo_encabezado = []


# === Preparación de funciones ===
x = symbols('x')
def exportar_csv(datos, encabezados):
    archivo = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Guardar como CSV"
    )
    if not archivo:
        return  # Cancelado por el usuario

    with open(archivo, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
        writer.writerows(datos)
def exportar_csv(datos, encabezados):
    archivo = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Guardar como CSV"
    )
    if not archivo:
        return  # Cancelado

    with open(archivo, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
        writer.writerows(datos)

# Función para exportar resultados a CSV
def exportar_resultado():
    if not ultimo_resultado or not ultimo_encabezado:
        messagebox.showinfo("Exportar", "No hay resultados disponibles para exportar.")
        return
    exportar_csv(ultimo_resultado, ultimo_encabezado)


# Función para buscar intervalos con cambio de signo
# Esta función evalúa la función en puntos sucesivos para encontrar intervalos donde hay un cambio de signo
def buscar_intervalos_con_signo(f, x_inicio, pasos=20):
    intervalos = []
    for i in range(pasos):
        x0 = x_inicio + i
        x1 = x0 + 1
        try:
            fx0 = f(x0)
            fx1 = f(x1)
            if np.sign(fx0) != np.sign(fx1):
                intervalos.append((x0, fx0, fx1))
        except Exception as e:
            print(f"Error evaluando en x={x0}: {e}")
            continue
    return intervalos



# === Funciones GUI ===
# Función para mostrar la tabla de resultados en el Text widget
# Esta función formatea los datos de la tabla y los muestra en el widget de salida
def mostrar_tabla(tabla, headers):
    output.delete("1.0", tk.END)
    output.insert(tk.END, "\t".join(headers) + "\n")
    for fila in tabla:
        output.insert(tk.END, "\t".join([f"{v:.6f}" if isinstance(v, float) else str(v) for v in fila]) + "\n")

# Función para leer la matriz A y el vector b desde los Text widgets
# Esta función evalúa el contenido de los widgets de texto y convierte las cadenas en matrices y vectores de NumPy
def leer_matriz_vector():
    try:
        A = eval(matrizA_text.get("1.0", "end").strip(), {"__builtins__": {}})
        b = eval(vectorb_text.get("1.0", "end").strip(), {"__builtins__": {}})
        return np.array(A, dtype=float), np.array(b, dtype=float)
    except Exception as e:
        messagebox.showerror("Error", f"Error al leer matriz/vector: {e}")
        return None, None
# Función para graficar el método de Gauss-Seidel
# Esta función toma el historial de iteraciones y grafica la convergencia de las variables    
def graficar_gauss_seidel(historial):
    historial = np.array(historial)
    num_vars = historial.shape[1]
    plt.figure(figsize=(8, 5))

    for i in range(num_vars):
        plt.plot(historial[:, i], label=f"x{i+1}")

    plt.title("Convergencia Gauss-Seidel")
    plt.xlabel("Iteración")
    plt.ylabel("Valor de las variables")
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para graficar el polinomio característico del método de Krylov
# Esta función toma los coeficientes del polinomio y grafica su comportamiento
# Utiliza NumPy para evaluar el polinomio en un rango de valores y Matplotlib
def graficar_polinomio_krylov(coeficientes):
    x = np.linspace(-10, 10, 400)
    y = np.polyval(list(reversed(coeficientes)), x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="Polinomio característico")
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Polinomio característico (Krylov)")
    plt.xlabel("λ")
    plt.ylabel("p(λ)")
    plt.grid(True)
    plt.legend()
    plt.show()
ultimo_resultado = []


# Función para exportar el resultado a CSV
# Esta función verifica si hay resultados disponibles y llama a la función de exportación
def exportar_resultado():
    if not ultimo_resultado:
        messagebox.showinfo("Exportar", "No hay resultados para exportar.")
        return
    exportar_csv(ultimo_resultado, ultimo_encabezado)

# Función principal que ejecuta el método seleccionado
# Esta función obtiene los parámetros de entrada, llama al método correspondiente y muestra los resultados    
def ejecutar():
    metodo = metodo_var.get()
    if metodo in ["Bisección", "Newton-Raphson", "Secante"]:
        expr = funcion_entry.get()
        f, f_prime, _ = parse_function(expr)
        if f is None:
            return
        
        # Vericamos que metodo se eligio
        if metodo == "Bisección":
            try:
                x_inicio = float(param1_entry.get())
            except:
                messagebox.showerror("Error", "Ingrese un valor válido.")
                return
            intervalos = buscar_intervalos_con_signo(f, x_inicio)
            if not intervalos:
                messagebox.showerror("Error", "No se encontró cambio de signo.")
                return
            output.delete("1.0", tk.END)
            output.insert(tk.END, "Cambios de signo encontrados:\n")
            output.insert(tk.END, "x\tf(x)\tf(x+1)\n")
            for x_val, fx, fx1 in intervalos:
                output.insert(tk.END, f"{x_val:.6f}\t{fx:.6f}\t{fx1:.6f}\n")
            a, b = intervalos[0][0], intervalos[0][0] + 1
            tabla, error = metodo_biseccion(f, a, b)
            if error:
                messagebox.showerror("Error", error)
                return
            output.insert(tk.END, "\nTabla de iteraciones de Bisección:\n")
            mostrar_tabla(tabla, ["Iteración", "a", "b", "n", "f(a)", "f(b)", "f(n)", "Error"])
            #EXPORTAMOS A CSV
            global ultimo_resultado, ultimo_encabezado
            ultimo_resultado = tabla  # la tabla generada
            ultimo_encabezado = ["x", "f(x)", "f(x+1)", "¿Cambio de signo?"]
        elif metodo == "Newton-Raphson":
            try:
                x0 = float(param1_entry.get())
            except:
                messagebox.showerror("Error", "Ingrese un valor válido.")
                return
            tabla, error = metodo_newton(f, f_prime, x0)
            if error:
                messagebox.showerror("Error", error)
                return
            mostrar_tabla(tabla, ["Iter", "x", "f(x)", "f'(x)", "x_nuevo", "Error"])
        
            # ================= GRAFICAR =================
            df = lambda x: (f(x + 1e-5) - f(x - 1e-5)) / (2e-5)
        
            def newton_raphson_graf(f, df, x0, tol=1e-6, max_iter=10):
                historial = [x0]
                for _ in range(max_iter):
                    try:
                        x1 = x0 - f(x0)/df(x0)
                    except ZeroDivisionError:
                        break
                    historial.append(x1)
                    if abs(x1 - x0) < tol:
                        break
                    x0 = x1
                return historial
        
            iteraciones = newton_raphson_graf(f, df, x0)
        
            x_vals = np.linspace(x0 - 10, x0 + 10, 400)
            y_vals = [f(x) for x in x_vals]
        
            plt.figure(figsize=(8, 5))
            plt.axhline(0, color="gray", linestyle="--")
            plt.plot(x_vals, y_vals, label="f(x)", color="blue")
        
            for i in range(len(iteraciones)-1):
                x = iteraciones[i]
                y = f(x)
                slope = df(x)
                tangent_x = np.linspace(x - 1, x + 1, 10)
                tangent_y = slope * (tangent_x - x) + y
                plt.plot(tangent_x, tangent_y, color="red", linestyle="--")
                plt.plot(x, y, 'ro')
        
            plt.title("Método de Newton-Raphson")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)
            plt.show()
        
            # ================= EXPORTAR =================
            ultimo_resultado = tabla
            ultimo_encabezado = ["Iteración", "x", "f(x)", "f'(x)", "x_nuevo", "Error"]

        elif metodo == "Secante":
            try:
                x0 = float(param1_entry.get())
                x1 = float(param2_entry.get())
            except:
                messagebox.showerror("Error", "Ingrese valores válidos.")
                return
            tabla, error = metodo_secante(f, x0, x1)
            if error:
                messagebox.showerror("Error", error)
                return
            mostrar_tabla(tabla, ["Iter", "x0", "x1", "f(x0)", "f(x1)", "x_nuevo"])

    elif metodo == "Gauss-Seidel":
        A, b = leer_matriz_vector()
        if A is None or b is None:
            return
        tabla = metodo_gauss_seidel(A, b)
        mostrar_tabla(tabla, ["Iter"] + [f"x{i+1}" for i in range(len(b))] + ["Error"])
        output.insert(tk.END, "\n\nVectores por iteración:\n")
        for fila in tabla:
            iteracion = fila[0]
            vector_x = fila[1:-1]  # quitamos Iter y Error
            output.insert(tk.END, f"Iteración {iteracion}: " + ", ".join(f"{val:.6f}" for val in vector_x) + "\n")  
        def graficar_gauss_seidel(historial):
            historial = np.array(historial)
            num_vars = historial.shape[1]
            plt.figure(figsize=(8, 5))
        
            for i in range(num_vars):
                plt.plot(historial[:, i], label=f"x{i+1}")
        
            plt.title("Convergencia Gauss-Seidel")
            plt.xlabel("Iteración")
            plt.ylabel("Valor de las variables")
            plt.legend()
            plt.grid(True)
            plt.show()
        graficar_gauss_seidel(tabla[:-1])  # Quitamos el último si es sólo error
        # Exportar a CSV
        ultimo_resultado = iteraciones  # matriz de resultados por iteración
        ultimo_encabezado = [f"x{i+1}" for i in range(len(iteraciones[0]))]
    elif metodo == "Krylov":
        A, b = leer_matriz_vector()
        if A is None or b is None:
            return
    
        coeficientes, vectores_krylov = metodo_krylov(A, b)
    
        output.delete("1.0", tk.END)
        output.insert(tk.END, "Coeficientes del polinomio característico:\n")
        output.insert(tk.END, str(coeficientes))
        output.insert(tk.END, "\n\nVectores de Krylov:\n")
        for i, vec in enumerate(vectores_krylov):
         output.insert(tk.END, f"v{i}: " + ", ".join(f"{val:.6f}" for val in vec) + "\n")
        def graficar_polinomio_krylov(coeficientes):
            x = np.linspace(-10, 10, 400)
            y = np.polyval(list(reversed(coeficientes)), x)
    
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, label="Polinomio característico")
            plt.axhline(0, color="gray", linestyle="--")
            plt.title("Polinomio característico (Krylov)")
            plt.xlabel("λ")
            plt.ylabel("p(λ)")
            plt.grid(True)
            plt.legend()
            plt.show()
    
        graficar_polinomio_krylov(coeficientes)
    
        # ========= Guardar para exportación =========
        ultimo_resultado = []
        for i, vec in enumerate(vectores_krylov):
            ultimo_resultado.append([f"v{i}"] + list(vec))
    
        ultimo_encabezado = ["Vector"] + [f"a{i}" for i in range(len(vectores_krylov[0]))]
        
# ============================= ACTUALIZAR CAMPOS DINÁMICAMENTE =============================
# Esta función actualiza los campos de entrada según el método seleccionado
# Se usa para mostrar u ocultar campos específicos según el método elegido
# Se conecta al evento de selección del combobox para actualizar los campos
        
def actualizar_campos(*args):
    metodo = metodo_var.get()
    for widget in [param1_label, param1_entry, param2_label, param2_entry,
                   funcion_label, funcion_entry, matrizA_label, matrizA_text,
                   vectorb_label, vectorb_text]:
        widget.grid_remove()

    if metodo in ["Bisección", "Newton-Raphson", "Secante"]:
        funcion_label.grid(row=2, column=0, sticky="w")
        funcion_entry.grid(row=2, column=1, sticky="w")
    if metodo == "Bisección":
        param1_label.config(text="Desde qué valor empezar búsqueda:")
        param1_label.grid(row=3, column=0, sticky="w")
        param1_entry.grid(row=3, column=1, sticky="w")
    elif metodo == "Newton-Raphson":
        param1_label.config(text="x0:")
        param1_label.grid(row=3, column=0, sticky="w")
        param1_entry.grid(row=3, column=1, sticky="w")
    elif metodo == "Secante":
        param1_label.config(text="x0:")
        param2_label.config(text="x1:")
        param1_label.grid(row=3, column=0, sticky="w")
        param1_entry.grid(row=3, column=1, sticky="w")
        param2_label.grid(row=4, column=0, sticky="w")
        param2_entry.grid(row=4, column=1, sticky="w")
    elif metodo in ["Gauss-Seidel", "Krylov"]:
        matrizA_label.grid(row=2, column=0, sticky="w")
        matrizA_text.grid(row=2, column=1)
        vectorb_label.grid(row=3, column=0, sticky="w")
        vectorb_text.grid(row=3, column=1)

# ============================= FUNCIONES AUXILIARES =============================
# Esta función convierte una expresión matemática en una función evaluable
# Utiliza sympy para convertir la cadena de texto en una expresión simbólica
def parse_function(expr):
    try:
        expr = expr.replace("^", "**")
        f_expr = sympify(expr)
        f = lambdify(x, f_expr, modules=['numpy'])
        f_prime = lambdify(x, diff(f_expr, x), modules=['numpy'])
        return f, f_prime, f_expr
    except Exception as e:
        messagebox.showerror("Error", f"Función inválida: {e}")
        return None, None, None

# ============================= UI MODERNA CON TTK STYLES =============================

# Crear la ventana principal y configurar estilos
root = tk.Tk()
root.title("Calculadora de Métodos Numéricos")
root.geometry("900x650")
root.configure(bg="#EAF6FF")

style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11), padding=6, background="#1E88E5", foreground="white")
style.configure("TLabel", font=("Segoe UI", 11), background="#EAF6FF")
style.configure("TEntry", font=("Segoe UI", 11))
style.configure("TCombobox", font=("Segoe UI", 11))

# ============================= FUNCIONES DE AYUDA =============================
# Esta función muestra un mensaje de ayuda según el método seleccionado
# Utiliza un diccionario para mapear cada método a su descripción
def mostrar_ayuda():
    metodo = metodo_var.get()
    mensaje = {
        "Bisección": (
            "Método de Bisección:\n\n"
            "Encuentra una raíz de la función en un intervalo [a, b] donde f(a) y f(b) tienen signos opuestos.\n"
            "Uso:\n- Ingresa la función.\n- Proporciona un valor desde el cual buscar el primer cambio de signo.\n"
            "Ejemplo de función: x**3 - x - 2"
        ),
        "Newton-Raphson": (
            "Método de Newton-Raphson:\n\n"
            "Usa derivadas para encontrar una raíz de la función, partiendo de una aproximación inicial.\n"
            "Uso:\n- Ingresa la función.\n- Proporciona un valor inicial x0.\n"
            "Ejemplo de función: x**3 - x - 2"
        ),
        "Secante": (
            "Método de la Secante:\n\n"
            "Similar al de Newton, pero sin usar derivadas. Utiliza dos valores iniciales para aproximar la raíz.\n"
            "Uso:\n- Ingresa la función.\n- Proporciona x0 y x1.\n"
            "Ejemplo de función: x**3 - x - 2"
        ),
        "Gauss-Seidel": (
            "Método de Gauss-Seidel:\n\n"
            "Resuelve un sistema de ecuaciones lineales Ax = b iterativamente.\n"
            "Uso:\n- Ingresa la matriz A y el vector b.\n"
            "Ejemplo:\nA = [[4,1],[2,3]]\nb = [1,2]"
        ),
        "Krylov": (
            "Método de Krylov:\n\n"
            "Calcula el polinomio característico de una matriz usando el método de potencias sucesivas.\n"
            "Uso:\n- Ingresa la matriz A y el vector b.\n"
            "Ejemplo:\nA = [[4,1],[2,3]]\nb = [1,2]"
        ),
    }.get(metodo, "Selecciona un método para ver la ayuda.")
    messagebox.showinfo("Ayuda", mensaje)

# ============================= ENCABEZADO =============================
 # Esta sección crea el encabezado de la aplicación con un título y un icono
root.iconbitmap("icono.ico")  # Asegúrate de tener un icono en el directorio
titulo = ttk.Label(root, text="Calculadora de Métodos Numéricos", font=("Segoe UI", 18, "bold"))
titulo.pack(pady=10)

frame_opciones = tk.Frame(root, bg="#EAF6FF")
frame_opciones.pack(pady=5)

metodo_var = tk.StringVar()

ttk.Label(frame_opciones, text="Selecciona el método:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
metodo_combo = ttk.Combobox(frame_opciones, textvariable=metodo_var, width=25,
    values=["Bisección", "Newton-Raphson", "Secante", "Gauss-Seidel", "Krylov"])
metodo_combo.grid(row=0, column=1, padx=5, pady=5)

ayuda_btn = ttk.Button(frame_opciones, text="¿Cómo usar?", command=mostrar_ayuda)
ayuda_btn.grid(row=0, column=2, padx=10)

# ============================= ENTRADAS DINÁMICAS =============================
frame_inputs = tk.Frame(root, bg="#EAF6FF")
frame_inputs.pack(pady=10)

# Etiquetas y entradas creadas dinámicamente
# Esta función crea etiquetas y entradas para los parámetros de cada método
# Se usa para evitar repetir código y facilitar la creación de campos según el método seleccionado
def crear_input(label_text):
    label = ttk.Label(frame_inputs, text=label_text)
    entry = ttk.Entry(frame_inputs, width=35)
    return label, entry

funcion_label, funcion_entry = crear_input("Función f(x):")
param1_label, param1_entry = crear_input("Parámetro 1:")
param2_label, param2_entry = crear_input("Parámetro 2:")

matrizA_label = ttk.Label(frame_inputs, text="Matriz A:")
matrizA_text = tk.Text(frame_inputs, height=4, width=40, font=("Consolas", 10))
vectorb_label = ttk.Label(frame_inputs, text="Vector b:")
vectorb_text = tk.Text(frame_inputs, height=2, width=40, font=("Consolas", 10))

# ============================= OUTPUT =============================
# Esta sección crea un widget de texto para mostrar los resultados
# Se usa para mostrar tablas, mensajes de error y resultados de los métodos
calcular_btn = ttk.Button(root, text="Calcular", command=ejecutar)
calcular_btn.pack(pady=10)

btn_exportar = ttk.Button(root, text="Exportar CSV", command=exportar_resultado)
btn_exportar.pack(pady=5)


output = tk.Text(root, height=15, width=100, font=("Consolas", 10), bg="#fdfdfd", bd=2, relief="sunken")
output.pack(pady=10)

# ============================= EVENTOS Y LOGICA DINÁMICA =============================
# Esta función se conecta al evento de selección del combobox para actualizar los campos de entrada
# Se usa para mostrar u ocultar campos específicos según el método elegido
def actualizar_campos(*args):
    for widget in frame_inputs.winfo_children():
        widget.grid_remove()
    metodo = metodo_var.get()
    row = 0
    if metodo in ["Bisección", "Newton-Raphson", "Secante"]:
        funcion_label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        funcion_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1
    if metodo == "Bisección":
        param1_label.config(text="Inicio de búsqueda:")
        param1_label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        param1_entry.grid(row=row, column=1, padx=5, pady=5)
    elif metodo == "Newton-Raphson":
        param1_label.config(text="x0:")
        param1_label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        param1_entry.grid(row=row, column=1, padx=5, pady=5)
    elif metodo == "Secante":
        param1_label.config(text="x0:")
        param2_label.config(text="x1:")
        param1_label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        param1_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        param2_label.grid(row=row, column=0, sticky="e", padx=5, pady=5)
        param2_entry.grid(row=row, column=1, padx=5, pady=5)
    elif metodo in ["Gauss-Seidel", "Krylov"]:
        matrizA_label.grid(row=row, column=0, sticky="ne", padx=5, pady=5)
        matrizA_text.grid(row=row, column=1, padx=5, pady=5)
        row += 1
        vectorb_label.grid(row=row, column=0, sticky="ne", padx=5, pady=5)
        vectorb_text.grid(row=row, column=1, padx=5, pady=5)

metodo_combo.bind("<<ComboboxSelected>>", actualizar_campos)

# ============================= MAINLOOP =============================
# Iniciar el bucle principal de la aplicación
root.mainloop()


