import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from kan import KAN
import os
import numpy as np

# Evitamos errores en Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# =========================================================
# 1. Nuestra "Realidad Oculta" (Las verdaderas f(x))
# =========================================================
# Usamos np.copy(x) para evitar el bug de las líneas discontinuas
funciones_f = {
    "f(x) = 0":   lambda x: np.zeros_like(x),
    "f(x) = 1":   lambda x: np.ones_like(x),
    "f(x) = x":   lambda x: np.copy(x),
    "f(x) = x^2": lambda x: x**2,
    "f(x) = x^4": lambda x: x**4
}

num_puntos = 100
x_np = np.linspace(0, 1, num_puntos)
h = x_np[1] - x_np[0] # Esto es nuestro dx matemático

# Generar datos de sensores (u real) mediante EDO numérica
A_mat = np.zeros((num_puntos, num_puntos))
A_mat[0, 0] = 1.0
A_mat[-1, -1] = 1.0
for i in range(1, num_puntos-1):
    A_mat[i, i-1] = -1.0 / h**2
    A_mat[i, i]   = (2.0 / h**2) + 1.0
    A_mat[i, i+1] = -1.0 / h**2

resultados_u_datos = {}
resultados_f_reconstruida = {}

print("Iniciando PROBLEMA INVERSO (Reconstrucción 1x2)...\n")

# =========================================================
# 2. Bucle del Problema Inverso
# =========================================================
for nombre, func_f in funciones_f.items():
    print(f"-> Descubriendo la física de: {nombre} ...", end=" ", flush=True)
    
    # Datos "experimentales" de u(x)
    f_vec = func_f(x_np)
    f_vec[0] = 0.0; f_vec[-1] = 0.0
    u_data = np.linalg.solve(A_mat, f_vec)
    resultados_u_datos[nombre] = u_data
    
    # Tensores para PyTorch
    x_train = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1)
    u_target = torch.tensor(u_data, dtype=torch.float32).unsqueeze(1)
    
    # EL "CORSÉ": grid=3 para suavizar la segunda derivada
    model = KAN(width=[1, 4, 1], grid=3, k=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs_inverso = 5000
    dx_tensor = x_train[1] - x_train[0] # Diferencial para la integral
    
    # Entrenamiento: La red ajusta la forma
    for epoch in range(epochs_inverso):
        optimizer.zero_grad()
        u_pred = model(x_train)
        
        # Loss usando el concepto de Integral (Sumatorio de errores * dx)
        loss = torch.sum(((u_pred - u_target)**2) * dx_tensor)
        
        loss.backward()
        optimizer.step()
        
    # Fase de Descubrimiento (Derivamos la red)
    x_eval = torch.linspace(0, 1, num_puntos).unsqueeze(1).requires_grad_(True)
    u_eval = model(x_eval)
    
    u_x = torch.autograd.grad(u_eval, x_eval, grad_outputs=torch.ones_like(u_eval), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_eval, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    # Aplicamos la ecuación para extraer f(x)
    f_descubierta = -u_xx + u_eval
    
    resultados_f_reconstruida[nombre] = f_descubierta.detach().numpy()
    print("[OK]")

# =========================================================
# 3. Generar la Matriz de Gráficas (1x2)
# =========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Problema Inverso: Descubriendo la Fuerza $f(x)$ a partir de $u(x)$", fontsize=16, y=0.95)

colores = ['gray', 'blue', 'green', 'orange', 'red']
estilos = ['-', '-', '--', '-.', ':']

# --- GRÁFICA IZQUIERDA: Los datos de entrada u(x) ---
for (nombre, u_val), color, estilo in zip(resultados_u_datos.items(), colores, estilos):
    ax1.plot(x_np, u_val, label=f"Sensores: {nombre}", color=color, linestyle=estilo, linewidth=2.5)

ax1.set_title("1. Datos Observados: Deformación $u(x)$", fontsize=14)
ax1.set_xlabel("Posición (x)", fontsize=12)
ax1.set_ylabel("Desplazamiento $u(x)$", fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=11, loc="upper left")

# --- GRÁFICA DERECHA: La f(x) Reconstruida por la Red ---
for (nombre, func_f), color, estilo in zip(funciones_f.items(), colores, estilos):
    f_real = func_f(x_np)
    # Dibujamos la real de fondo gruesa
    ax1.plot([], [], color=color, alpha=0.3, linewidth=5) # Truco para la leyenda si hace falta
    ax2.plot(x_np, f_real, color=color, alpha=0.3, linewidth=6)
    
    # Dibujamos la que ha descubierto la KAN
    f_red = resultados_f_reconstruida[nombre]
    ax2.plot(x_np, f_red, label=f"IA Descubre {nombre}", color=color, linestyle=estilo, linewidth=2)

ax2.set_title("2. Física Descubierta: $f(x)$ Reconstruida", fontsize=14)
ax2.set_xlabel("Posición (x)", fontsize=12)
ax2.set_ylabel("Magnitud de la fuerza $f(x)$", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=11, loc="upper left")

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.show()
