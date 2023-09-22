import torch.nn as nn
import torch

# Crea una clase de modelo de regresión lineal
class ModeloRegresionLineal(nn.Module):
  def __init__(self):
    super().__init__()
    self.volumen = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    self.sesgo = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

# Define el cálculo en el modelo
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.volumen * x + self.sesgo 