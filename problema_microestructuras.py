import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

# ==========================================
# 1. Configuración del Problema
# ==========================================
# Lista de tuplas: (epsilon, numero_de_puntos)
configuraciones = [
    (1.0, 100),     # Epsilon 1.0 -> 100 puntos
    (0.1, 100),     # Epsilon 0.1 -> 100 puntos
    (0.01, 1000),   # Epsilon 0.01 -> 1000 puntos
    (0.001, 10000)  # Epsilon 0.001 -> 10000 puntos
]

T_final = 1.0  
t_eval = 0.5   # Instante de tiempo que usaremos para la gráfica final

class MicroPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitectura MLP un poco más ancha para captar altas frecuencias
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        salida = self.net(inputs)
        # ANSATZ: Obliga matemáticamente a cumplir u(x,0)=sin(pi*x) y u(0,t)=u(1,t)=0
        u = torch.sin(torch.pi * x) + t * x * (1.0 - x) * salida
        return u

resultados_u = {}

# ==========================================
# 2. Bucle de Entrenamiento
# ==========================================
print("Iniciando estudio de Homogeneización de Microestructuras...\n")

for epsilon, n_puntos in configuraciones:
    print(f"-> Entrenando para epsilon = {epsilon} (con {n_puntos} puntos)...")
    
    model = MicroPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Generamos los puntos base FUERA del bucle
    x_base = torch.rand(n_puntos, 1)
    t_base = T_final * torch.rand(n_puntos, 1)
    
    epochs = 1500
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ==============================================================
        # EL TRUCO MAGISTRAL: Clonamos para evitar el error del grafo
        # ==============================================================
        x_colloc = x_base.clone().requires_grad_(True)
        t_colloc = t_base.clone().requires_grad_(True)
        
        u = model(x_colloc, t_colloc)
        
        u_t = torch.autograd.grad(u, t_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x_colloc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_colloc, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # Ecuación de la Microestructura
        coeficiente = 10.0 * (2.0 + torch.sin(torch.pi * x_colloc / epsilon))
        residuo = coeficiente * u_t - u_xx
        
        loss = torch.mean(residuo**2)
        loss.backward()
        optimizer.step()
        
    print(f"   [OK] Loss final: {loss.item():.4e}\n")
    
    # Guardamos la predicción en t=0.5 para graficar luego
    x_plot = np.linspace(0, 1, 300).reshape(-1, 1)
    x_tensor = torch.tensor(x_plot, dtype=torch.float32)
    t_tensor = torch.full_like(x_tensor, t_eval)
    
    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).numpy()
    
    resultados_u[epsilon] = u_pred
    
# ==========================================
# 3. Gráfica 2x2 de Resultados
# ==========================================
print("Generando matriz de resultados...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Homogeneización de la Temperatura (Evaluado en $t={t_eval}$)", fontsize=16)

x_plot_flat = x_plot.flatten()

for i, (epsilon, _) in enumerate(configuraciones):
    ax = axes[i // 2, i % 2]
    u_val = resultados_u[epsilon]
    
    ax.plot(x_plot_flat, u_val, 'r-', linewidth=2.5, label=f"PINN ($\epsilon={epsilon}$)")
    ax.set_title(f"Parámetro microestructural $\epsilon = {epsilon}$ | Puntos: {configuraciones[i][1]}")
    ax.set_xlabel("Posición (x)")
    ax.set_ylabel("Temperatura u(x,t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1) 
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

 

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Parámetros del experimento
# ==========================================

eps_list = [1, 0.1, 0.01]
points_list = [100, 1800, 15000]

T_final = 1.0

# ==========================================
# Red neuronal
# ==========================================

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,1)
        )
        
    def forward(self,x,t):
        X = torch.cat([x,t],1)
        return self.net(X)

# ==========================================
# Derivadas
# ==========================================

def grad(y,x):
    return torch.autograd.grad(
        y,x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

# ==========================================
# Condición inicial
# ==========================================

def u0(x):
    return torch.sin(np.pi*x)

# ==========================================
# Entrenamiento para cada epsilon
# ==========================================

results = {}

for eps, N in zip(eps_list, points_list):

    print("\n============================")
    print(f"Entrenando para epsilon = {eps}")
    print("============================")

    net = PINN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)

    epochs = 3000

    for epoch in range(epochs):

        opt.zero_grad()

        # ------------------
        # Collocation points
        # ------------------

        x = torch.rand(N,1,device=device,requires_grad=True)
        t = T_final*torch.rand(N,1,device=device,requires_grad=True)

        u = net(x,t)

        u_t = grad(u,t)
        u_x = grad(u,x)
        u_xx = grad(u_x,x)

        coef = 10*(2 + torch.sin(np.pi*x/eps))

        pde = torch.mean((coef*u_t - u_xx)**2)

        # ------------------
        # Initial condition
        # ------------------

        x0 = torch.rand(200,1,device=device,requires_grad=True)
        t0 = torch.zeros_like(x0)

        ic = torch.mean((net(x0,t0) - u0(x0))**2)

        # ------------------
        # Boundary condition
        # ------------------

        tb = T_final*torch.rand(200,1,device=device)

        xb1 = torch.zeros_like(tb, requires_grad=True)
        xb2 = torch.ones_like(tb, requires_grad=True)

        bc = torch.mean(net(xb1,tb)**2 + net(xb2,tb)**2)

        loss = pde + 20*ic + 20*bc

        loss.backward()
        opt.step()

        if epoch % 500 == 0:
            print(epoch, loss.item())

    # ==========================================
    # Evaluación
    # ==========================================

    x_plot = np.linspace(0,1,400)
    t_plot = 0.5*np.ones_like(x_plot)

    X = torch.tensor(x_plot.reshape(-1,1),dtype=torch.float32).to(device)
    T = torch.tensor(t_plot.reshape(-1,1),dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred = net(X,T).cpu().numpy()

    results[eps] = u_pred

# ==========================================
# REPRESENTACIONES
# ==========================================

plt.figure(figsize=(8,5))

for eps in eps_list:
    plt.plot(x_plot, results[eps], label=f"ε={eps}")

plt.title("Solución PINN para distintos ε")
plt.xlabel("x")
plt.ylabel("u(x,t=0.5)")
plt.legend()
plt.grid(True)
plt.savefig("archivo.png", dpi=300)
plt.close()
