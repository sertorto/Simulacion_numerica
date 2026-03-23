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
# Condiciones de borde: u(0) = 0, u(1) = 0
casos = [
   {"nombre": "1",   "F": lambda x: torch.ones_like(x)},
   {"nombre": "x",   "F": lambda x: x},
   {"nombre": "x^2", "F": lambda x: x**2},
   {"nombre": "x^3", "F": lambda x: x**3}
]


# Soluciones analíticas recalculadas para el dominio [0, 1]
def solucion_exacta(x_np, caso_nombre):
   sh1 = np.sinh(1.0)
   ch1 = np.cosh(1.0)
  
   if caso_nombre == "1":
       return 1.0 - np.cosh(x_np) + ((ch1 - 1.0) / sh1) * np.sinh(x_np)
   elif caso_nombre == "x":
       return x_np - np.sinh(x_np) / sh1
   elif caso_nombre == "x^2":
       return x_np**2 + 2.0 - 2.0 * np.cosh(x_np) + ((2.0 * ch1 - 3.0) / sh1) * np.sinh(x_np)
   elif caso_nombre == "x^3":
       return x_np**3 + 6.0 * x_np - 7.0 * np.sinh(x_np) / sh1


# ===============================
# 3. PREPARACIÓN DE DATOS (Dominio [0, 1])
# ===============================
N_physics = 300
# CAMBIO CLAVE: Dominio de 0 a 1
x_phys = torch.linspace(0, 1, N_physics).view(-1, 1).requires_grad_(True)


# Puntos de frontera en x=0 y x=1
x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
u_bc_target = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# ===============================
# 4. ENTRENAMIENTO SECUENCIAL
# ===============================
num_epochs = 1500
modelos_entrenados = []


print("Entrenando las 4 redes PIKAN (Dominio [0, 1])...")


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
      
       # Soft constraint para bordes en 0 y 1
       u_bc_pred = modelo(x_bc)
       loss_bc = torch.mean((u_bc_pred - u_bc_target)**2)
      
       # Pérdida total (se le da mucho peso a los bordes para clavar el (0,0) y (1,0))
       loss = loss_pde + 10.0 * loss_bc
       loss.backward()
       optimizador.step()
      
   modelos_entrenados.append(modelo)


print("\n¡Entrenamientos completados! Generando tabla 4x2...")


# ===============================
# 5. GENERACIÓN DE LA TABLA 4x2
# ===============================
# CAMBIO CLAVE: Malla de visualización adaptada a [0, 1]
x_plot_np = np.linspace(0, 1, 200)
x_plot_tensor = torch.tensor(x_plot_np, dtype=torch.float32).view(-1, 1)


fig, axs = plt.subplots(4, 2, figsize=(10, 14))
fig.suptitle('PIKAN para $-u_{xx} + u = F(x)$ | $u(0)=0, u(1)=0$', fontsize=16, fontweight='bold', y=0.98)


with torch.no_grad():
   for i, caso in enumerate(casos):
       # Evaluar F(x)
       F_x_tensor = caso["F"](x_plot_tensor)
       F_x_np = F_x_tensor.numpy()
      
       # Evaluar u(x) predicha y exacta
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
   # Ajustamos los límites de la gráfica al nuevo dominio
   ax.set_xlim([0, 1])
  
plt.tight_layout(rect=[0, 0.02, 1, 0.96])


save_path = "/Users/molab/Documents/simnu/Tabla_PIKAN_F_x_dominio_0_1.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"Imagen estática guardada con éxito en:\n{save_path}")
plt.show()
