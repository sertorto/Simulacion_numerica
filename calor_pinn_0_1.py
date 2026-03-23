import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ===============================
# 1. ARQUITECTURA PINN ESTÁNDAR (MLP)
# ===============================
class PINN_MLP(nn.Module):
   def __init__(self, num_neurons=20):
       super().__init__()
       # Red Neuronal Multicapa (MLP) clásica para 1D
       # Entrada: 1 (x), Salida: 1 (u)
       # Usamos 3 capas ocultas con función de activación Tanh
       self.net = nn.Sequential(
           nn.Linear(1, num_neurons),
           nn.Tanh(),
           nn.Linear(num_neurons, num_neurons),
           nn.Tanh(),
           nn.Linear(num_neurons, num_neurons),
           nn.Tanh(),
           nn.Linear(num_neurons, 1)
       )
  
   def forward(self, x):
       return self.net(x)


# ===============================
# 2. DEFINICIÓN DE LOS CASOS (F(x) y Soluciones Exactas)
# ===============================
# Ecuación: -u_xx + u = F(x) | Condiciones: u(0) = 0, u(1) = 0
casos = [
   {"nombre": "1",   "F": lambda x: torch.ones_like(x)},
   {"nombre": "x",   "F": lambda x: x},
   {"nombre": "x^2", "F": lambda x: x**2},
   {"nombre": "x^3", "F": lambda x: x**3}
]


# Soluciones analíticas exactas en el dominio [0, 1]
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
x_phys = torch.linspace(0, 1, N_physics).view(-1, 1).requires_grad_(True)


# Puntos de frontera (Soft Constraints)
x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
u_bc_target = torch.tensor([[0.0], [0.0]], dtype=torch.float32)


# ===============================
# 4. ENTRENAMIENTO SECUENCIAL (PINN MLP)
# ===============================
num_epochs = 2000 # Le damos unas cuantas épocas más, los MLP a veces tardan un poco más en converger
modelos_entrenados = []


print("Entrenando las 4 redes PINN estándar (MLP)...")


for i, caso in enumerate(casos):
   print(f"-> Entrenando Caso {i+1}/4: F(x) = {caso['nombre']}")
   modelo = PINN_MLP(num_neurons=20)
   # Tasa de aprendizaje típica para MLPs en problemas sencillos
   optimizador = optim.Adam(modelo.parameters(), lr=0.005)
   F_funcion = caso["F"]
  
   for epoch in range(num_epochs):
       optimizador.zero_grad()
      
       # Predicción
       u_pred = modelo(x_phys)
      
       # Diferenciación Automática (Autograd)
       u_x = torch.autograd.grad(u_pred, x_phys, torch.ones_like(u_pred), create_graph=True)[0]
       u_xx = torch.autograd.grad(u_x, x_phys, torch.ones_like(u_x), create_graph=True)[0]
      
       # Residuo PDE: -u_xx + u - F(x) = 0
       F_x = F_funcion(x_phys)
       res_pde = -u_xx + u_pred - F_x
       loss_pde = torch.mean(res_pde**2)
      
       # Soft constraint para bordes
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
x_plot_np = np.linspace(0, 1, 200)
x_plot_tensor = torch.tensor(x_plot_np, dtype=torch.float32).view(-1, 1)


fig, axs = plt.subplots(4, 2, figsize=(10, 14))
fig.suptitle('PINN (MLP) para $-u_{xx} + u = F(x)$ | $u(0)=0, u(1)=0$', fontsize=16, fontweight='bold', y=0.98)


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
       axs[i, 1].plot(x_plot_np, u_pred, 'b--', lw=2.5, label='PINN (MLP)')
       axs[i, 1].set_title(f'Solución $u(x)$ para $F(x) = {caso["nombre"]}$')
       axs[i, 1].set_ylabel('$u(x)$')
       axs[i, 1].grid(True, linestyle='--', alpha=0.6)
      
       if i == 0:
           axs[i, 1].legend()


# Ajuste estético final
for ax in axs.flatten():
   ax.set_xlabel('$x$')
   ax.set_xlim([0, 1])
  
plt.tight_layout(rect=[0, 0.02, 1, 0.96])


save_path = "/Users/molab/Documents/simnu/Tabla_PINN_MLP_F_x.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"Imagen estática guardada con éxito en:\n{save_path}")
plt.show()
