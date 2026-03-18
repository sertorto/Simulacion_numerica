import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===============================
# 1. PROBLEMA
# ===============================
# y' = y
# y(0) = 1
# Solución exacta: exp(x)

epochs = 2000

# ===============================
# 2. ARQUITECTURAS
# ===============================

class KANLayer(nn.Module):
    def __init__(self, input_size, output_size, degree):
        super().__init__()
        self.degree = degree
        self.weights = nn.Parameter(torch.randn(output_size, input_size*(degree+1))*0.5)
        self.bias = nn.Parameter(torch.zeros(output_size,1))

    def forward(self,x):
        Phi = [x**d for d in range(self.degree+1)]
        Phi = torch.cat(Phi, dim=1)
        return Phi @ self.weights.T + self.bias.T


class KAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = KANLayer(1,20,3)
        self.k2 = KANLayer(20,20,3)
        self.out = nn.Linear(20,1)

    def forward(self,x):
        x = torch.tanh(self.k1(x))
        x = torch.tanh(self.k2(x))
        return self.out(x)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,20), nn.Tanh(),
            nn.Linear(20,20), nn.Tanh(),
            nn.Linear(20,20), nn.Tanh(),
            nn.Linear(20,1)
        )

    def forward(self,x):
        return self.net(x)

# ===============================
# 3. DATOS
# ===============================

N = 200
x_phys = torch.linspace(0,2,N).reshape(-1,1).requires_grad_(True)

x_ic = torch.tensor([[0.0]])
y_ic = torch.tensor([[1.0]])

x_plot = torch.linspace(0,2,200).reshape(-1,1)
y_exact = torch.exp(x_plot)

# ===============================
# 4. MODELOS
# ===============================

model_kan = KAN()
model_pinn = PINN()

opt_kan = optim.Adam(model_kan.parameters(), lr=0.01)
opt_pinn = optim.Adam(model_pinn.parameters(), lr=0.01)

# ===============================
# 5. FUNCIÓN DE PÉRDIDA
# ===============================

def loss_fn(model):

    y = model(x_phys)

    dy_dx = torch.autograd.grad(y, x_phys,
                                grad_outputs=torch.ones_like(y),
                                create_graph=True)[0]

    loss_ode = torch.mean((dy_dx - y)**2)

    loss_ic = (model(x_ic) - y_ic)**2

    return loss_ode + loss_ic

# ===============================
# 6. ENTRENAMIENTO + GIF
# ===============================

loss_kan_hist = []
loss_pinn_hist = []
frames = []

plt.ioff()
fig, (ax_loss, ax_sol) = plt.subplots(1,2, figsize=(12,5))

for epoch in range(1, epochs+1):

    # KAN
    opt_kan.zero_grad()
    l_kan = loss_fn(model_kan)
    l_kan.backward()
    opt_kan.step()
    loss_kan_hist.append(l_kan.item())

    # PINN
    opt_pinn.zero_grad()
    l_pinn = loss_fn(model_pinn)
    l_pinn.backward()
    opt_pinn.step()
    loss_pinn_hist.append(l_pinn.item())

    if epoch % 20 == 0:

        print(f"Epoch {epoch} | KAN: {l_kan.item():.6f} | PINN: {l_pinn.item():.6f}")

        ax_loss.clear()
        ax_loss.plot(loss_kan_hist,'r',label='KAN')
        ax_loss.plot(loss_pinn_hist,'b',label='PINN')
        ax_loss.set_yscale('log')
        ax_loss.legend()
        ax_loss.set_title("Evolución de la pérdida")

        y_kan = model_kan(x_plot).detach()
        y_pinn = model_pinn(x_plot).detach()

        ax_sol.clear()
        ax_sol.plot(x_plot, y_exact, 'k--', label='Exacta')
        ax_sol.plot(x_plot, y_kan, 'r', label='KAN')
        ax_sol.plot(x_plot, y_pinn, 'b', label='PINN')
        ax_sol.set_title("Aproximación")
        ax_sol.legend()

        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(image.copy())

# ===============================
# 7. GUARDAR GIF
# ===============================

save_path = os.path.join(os.getcwd(), "ODE_PINN_vs_KAN.gif")
imageio.mimsave(save_path, frames, fps=5, loop=0)

plt.close(fig)

print("\nGIF guardado en:", save_path)
print("Frames totales:", len(frames))
