import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.integrate import solve_ivp

# ===============================
# 1️⃣ Definir capa KAN
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
        y = Phi @ self.weights.T + self.bias.T
        return y

# ===============================
# 2️⃣ Red KAN
# ===============================
class KANNetwork(nn.Module):
    def __init__(self, num_neurons=12, degree=6):
        super().__init__()
        self.kan1 = KANLayer(1, num_neurons, degree)
        self.kan2 = KANLayer(num_neurons, num_neurons, degree)
        self.kan3 = KANLayer(num_neurons, num_neurons, degree)
        self.kan4 = KANLayer(num_neurons, num_neurons, degree)
        self.out = nn.Linear(num_neurons, 1)
    
    def forward(self, x):
        x = torch.tanh(self.kan1(x))
        x = torch.tanh(self.kan2(x))
        x = torch.tanh(self.kan3(x))
        x = torch.tanh(self.kan4(x))
        x = self.out(x)
        return x

# ===============================
# 2️⃣a Red PINN clásica
# ===============================
class PINNNetwork(nn.Module):
    def __init__(self, num_neurons=12, num_layers=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, num_neurons))
        layers.append(nn.Tanh())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(num_neurons,1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ===============================
# 3️⃣ Datos en [0,2]
# ===============================
x_train = torch.linspace(0, 2, 500).reshape(-1,1)
x_kan = x_train.clone().detach().requires_grad_(True)
x_pinn = x_train.clone().detach().requires_grad_(True)

x0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
y0_target = torch.tensor([[1.0]], dtype=torch.float32)

# ===============================
# 4️⃣ Solución de referencia
# ===============================
def ode_func(x, y):
    return 1 - y / (2 + 1.5*np.sin(4*np.pi*x))

sol_ref = solve_ivp(
    ode_func,
    [0,2],
    [1],
    t_eval=x_train.numpy().flatten(),
    method='DOP853',
    rtol=1e-9,
    atol=1e-12
)
y_ref = sol_ref.y[0]

# ===============================
# 5️⃣ Crear redes y optimizadores
# ===============================
num_neurons = 12

net_kan = KANNetwork(num_neurons=num_neurons, degree=6)
optimizer_kan = optim.Adam(net_kan.parameters(), lr=0.001)

pinn_net = PINNNetwork(num_neurons=num_neurons, num_layers=6)
optimizer_pinn = optim.Adam(pinn_net.parameters(), lr=0.005)

num_epochs = 3000
loss_kan_history = []
loss_pinn_history = []

# ===============================
# 6️⃣ Función para derivadas
# ===============================
def gradients(y, x, order=1):
    for _ in range(order):
        dy = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                 create_graph=True, retain_graph=True)[0]
        y = dy
    return y

# ===============================
# 7️⃣ Configurar figuras
# ===============================
plt.ioff()
fig, (ax_loss, ax_sol) = plt.subplots(1,2, figsize=(12,5))

ax_loss.set_xlim(0, num_epochs)
ax_loss.set_ylim(1e-4, 10)   # escala log
ax_loss.set_yscale("log")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Evolución de la pérdida (escala log)")
ax_loss.grid(True)
line_loss_kan, = ax_loss.plot([], [], 'b-', label='KAN Loss')
line_loss_pinn, = ax_loss.plot([], [], 'r-', label='PINN Loss')
ax_loss.legend()

ax_sol.set_xlim(0,2)
ax_sol.set_xticks(np.arange(0, 2.1, 0.5))
ax_sol.set_ylim(0.8, 1.4)
line_sol_kan, = ax_sol.plot([], [], 'b--', label='KAN')
line_sol_pinn, = ax_sol.plot([], [], 'r--', label='PINN')
ax_sol.plot(x_train.numpy(), y_ref, 'k-', label='Referencia')
ax_sol.set_title("Aproximación de la solución")
ax_sol.grid(True)
ax_sol.legend()

frames = []
frame_interval = 20  # capturar cada 20 epochs

# ===============================
# 8️⃣ Entrenamiento
# ===============================
eps = 1e-12  # para log-safe

for epoch in range(1, num_epochs+1):
    # ----- KAN -----
    optimizer_kan.zero_grad()
    y_pred_kan = net_kan(x_kan)
    dy_dx_kan = gradients(y_pred_kan, x_kan)
    f_kan = dy_dx_kan - (1 - y_pred_kan / (2 + 1.5*torch.sin(4*np.pi*x_kan)))
    loss_eq_kan = torch.mean(f_kan**2)
    loss_bc_kan = (net_kan(x0) - y0_target)**2
    loss_kan = loss_eq_kan + loss_bc_kan
    loss_kan.backward(retain_graph=True)
    optimizer_kan.step()
    loss_kan_history.append(loss_kan.item() + eps)

    # ----- PINN -----
    optimizer_pinn.zero_grad()
    y_pred_pinn = pinn_net(x_pinn)
    dy_dx_pinn = gradients(y_pred_pinn, x_pinn)
    f_pinn = dy_dx_pinn - (1 - y_pred_pinn / (2 + 1.5*torch.sin(4*np.pi*x_pinn)))
    loss_eq_pinn = torch.mean(f_pinn**2)
    loss_bc_pinn = (pinn_net(x0) - y0_target)**2
    loss_pinn = loss_eq_pinn + loss_bc_pinn
    loss_pinn.backward()
    optimizer_pinn.step()
    loss_pinn_history.append(loss_pinn.item() + eps)

    # Captura para GIF
    if epoch % frame_interval == 0 or epoch == 1:
        print(f"Epoch {epoch}, Loss KAN={loss_kan.item():.6e}, Loss PINN={loss_pinn.item():.6e}")

        with torch.no_grad():
            y_plot_kan = net_kan(x_train).detach().numpy()
            y_plot_pinn = pinn_net(x_train).detach().numpy()

        # Actualizar líneas
        line_sol_kan.set_data(x_train.numpy(), y_plot_kan)
        line_sol_pinn.set_data(x_train.numpy(), y_plot_pinn)
        line_loss_kan.set_data(range(len(loss_kan_history)), loss_kan_history)
        line_loss_pinn.set_data(range(len(loss_pinn_history)), loss_pinn_history)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

# ===============================
# 9️⃣ Mostrar resultado final
# ===============================
plt.ion()
line_sol_kan.set_data(x_train.numpy(), net_kan(x_train).detach().numpy())
line_sol_pinn.set_data(x_train.numpy(), pinn_net(x_train).detach().numpy())
plt.show()

# ===============================
# 🔟 Guardar GIF
# ===============================
imageio.mimsave("KAN_vs_PINN_solution.gif", frames, fps=10)
print("GIF guardado correctamente.")
