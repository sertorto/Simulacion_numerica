import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. ARQUITECTURA KAN (PIKAN 1D)
# ===============================
class KANLayer(nn.Module):
   def __init__(self, input_size, output_size, degree):
       super().__init__()
       self.input_size = input_size
       self.output_size = output_size
       self.degree = degree
       self.weights = nn.Parameter(torch.randn(output_size, input_size*(degree+1)) * 0.5)
       self.bias = nn.Parameter(torch.zeros(output_size, 1))
  
   def forward(self, x):
       Phi = [x**d for d in range(self.degree+1)]
       Phi = torch.cat(Phi, dim=1)
       return Phi @ self.weights.T + self.bias.T


class KANNetwork1D(nn.Module):
   def __init__(self, num_neurons=10, degree=3):
       super().__init__()
       # Entrada 1D (x), Salida 1D (u)
       self.kan1 = KANLayer(1, num_neurons, degree)
       self.kan2 = KANLayer(num_neurons, num_neurons, degree)
       self.out = nn.Linear(num_neurons, 1)
  
   def forward(self, x):
       x = torch.tanh(self.kan1(x))
       x = torch.tanh(self.kan2(x))
       return self.out(x)


# ===============================
# 2. DEFINICIÓN DE LOS CASOS (F(x) y Soluciones Exactas)
# ===============================
# Condiciones de borde: u(-1) = 0, u(1) = 0
casos = [
   {"nombre": "1",   "F": lambda x: torch.ones_like(x)},
   {"nombre": "x",   "F": lambda x: x},
   {"nombre": "x^2", "F": lambda x: x**2},
   {"nombre": "x^3", "F": lambda x: x**3}
]


# Funciones exactas analíticas para comparar y validar que la red aprende la física
def solucion_exacta(x_np, caso_nombre):
   if caso_nombre == "1":
       return 1.0 - np.cosh(x_np) / np.cosh(1.0)
   elif caso_nombre == "x":
       return x_np - np.sinh(x_np) / np.sinh(1.0)
   elif caso_nombre == "x^2":
       return x_np**2 + 2.0 - 3.0 * np.cosh(x_np) / np.cosh(1.0)
   elif caso_nombre == "x^3":
       return x_np**3 + 6.0 * x_np - 7.0 * np.sinh(x_np) / np.sinh(1.0)


# ===============================
# 3. PREPARACIÓN DE DATOS (Soft Constraints)
# ===============================
N_physics = 300
x_phys = torch.linspace(-1, 1, N_physics).view(-1, 1).requires_grad_(True)


# Puntos de frontera
x_bc = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
u_bc_target = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# ===============================
# 4. ENTRENAMIENTO SECUENCIAL DE LOS 4 MODELOS
# ===============================
num_epochs = 1500
modelos_entrenados = []


print("Entrenando las 4 redes PIKAN (una por cada F(x))...")


for i, caso in enumerate(casos):
   print(f"-> Entrenando Caso {i+1}/4: F(x) = {caso['nombre']}")
   modelo = KANNetwork1D()
   optimizador = optim.Adam(modelo.parameters(), lr=0.01)
   F_funcion = caso["F"]
  
   for epoch in range(num_epochs):
       optimizador.zero_grad()
      
       # Predicción
       u_pred = modelo(x_phys)
      
       # Derivadas (Autograd)
       u_x = torch.autograd.grad(u_pred, x_phys, torch.ones_like(u_pred), create_graph=True)[0]
       u_xx = torch.autograd.grad(u_x, x_phys, torch.ones_like(u_x), create_graph=True)[0]
      
       # Residuo PDE: -u_xx + u - F(x) = 0
       F_x = F_funcion(x_phys)
       res_pde = -u_xx + u_pred - F_x
       loss_pde = torch.mean(res_pde**2)
      
       # Soft constraint para bordes (MSE estándar)
       u_bc_pred = modelo(x_bc)
       loss_bc = torch.mean((u_bc_pred - u_bc_target)**2)
      
       # Pérdida total
       loss = loss_pde + 10.0 * loss_bc
       loss.backward()
       optimizador.step()
      
   modelos_entrenados.append(modelo)


print("\n¡Entrenamientos completados! Generando tabla 4x2...")


# ===============================
# 5. GENERACIÓN DE LA TABLA 4x2
# ===============================
x_plot_np = np.linspace(-1, 1, 200)
x_plot_tensor = torch.tensor(x_plot_np, dtype=torch.float32).view(-1, 1)


fig, axs = plt.subplots(4, 2, figsize=(10, 14))
fig.suptitle('PIKAN para $-u_{xx} + u = F(x)$ | $u(-1)=0, u(1)=0$', fontsize=16, fontweight='bold', y=0.98)


with torch.no_grad():
   for i, caso in enumerate(casos):
       F_x_tensor = caso["F"](x_plot_tensor)
       F_x_np = F_x_tensor.numpy()
      
       u_pred = modelos_entrenados[i](x_plot_tensor).numpy()
       u_exact = solucion_exacta(x_plot_np, caso["nombre"])
      
       # COLUMNA 1: Gráfica de F(x)
       axs[i, 0].plot(x_plot_np, F_x_np, 'g-', lw=2.5)
       axs[i, 0].set_title(f'$F(x) = {caso["nombre"]}$')
       axs[i, 0].set_ylabel('$F(x)$')
       axs[i, 0].grid(True, linestyle='--', alpha=0.6)
      
       # COLUMNA 2: Gráfica de u(x)
       axs[i, 1].plot(x_plot_np, u_exact, 'k-', lw=4, alpha=0.3, label='Exacta Analítica')
       axs[i, 1].plot(x_plot_np, u_pred, 'r--', lw=2.5, label='PIKAN')
       axs[i, 1].set_title(f'Solución $u(x)$ para $F(x) = {caso["nombre"]}$')
       axs[i, 1].set_ylabel('$u(x)$')
       axs[i, 1].grid(True, linestyle='--', alpha=0.6)
      
       if i == 0:
           axs[i, 1].legend()


# Ajuste estético final
for ax in axs.flatten():
   ax.set_xlabel('$x$')
  
plt.tight_layout(rect=[0, 0.02, 1, 0.96])


save_path = "/Users/molab/Documents/simnu/Tabla_PIKAN_F_x.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"Imagen estática guardada con éxito en:\n{save_path}")
plt.show()
