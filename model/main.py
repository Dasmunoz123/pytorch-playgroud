import torch
from torch import nn
from utils import plot_predictions
from classes import ModeloRegresionLineal
import matplotlib.pyplot as plt

## Check pytorch version
print(torch.__version__)


# Crea *nuevos* parámetros
volumen = 0.8
sesgo = 0.2

# Crea datos
inicio = 0
final = 1
step = 0.025
X = torch.arange(start=inicio, end=final, step=step).unsqueeze(dim=1)
y = volumen * X + sesgo

"""
'Creating training 6 testing splitting
"""
train_division = int(0.7 * len(X))
X_ent, y_ent = X[:train_division], y[:train_division]
X_prueb, y_prueb = X[train_division:], y[train_division:]

plot_predictions(datos_ent=X_ent, etiq_ent=y_ent, datos_prueba=X_prueb,  etiq_prueba=y_prueb, predictions=None, name='base')

## Modelo
torch.manual_seed(seed=42)
model_a = ModeloRegresionLineal()
print(model_a.state_dict())

## Predictions with no training
with torch.inference_mode():
    y_inference = model_a(X_prueb)



# Crea función de pérdida
fn_perd = torch.nn.L1Loss()

# Crea el optimizador
# tasa de aprendizaje (cuánto debe cambiar el optimizador de parámetros en cada paso, más alto = más (menos estable), más bajo = menos (puede llevar mucho tiempo))
optimizador = torch.optim.SGD(params=model_a.parameters(), lr=0.01)  

torch.manual_seed(42)

# Establezca cuántas veces el modelo pasará por los datos de entrenamiento
epocas = 300

# Cree listas de vacías para realizar un seguimiento de nuestro modelo
entrenamiento_loss = []
test_loss = []

for epoca in range(epocas):
    ### Entrenamiento

    # Pon el modelo en modo entrenamiento
    model_a.train()
    
    # 1. Pase hacia adelante los datos usando el método forward() 
    y_predc = model_a(X_ent)

    # 2. Calcula la pérdida (Cuán diferentes son las predicciones de nuestros modelos)
    perdida = fn_perd(y_predc, y_ent)

    # 3. Gradiente cero del optomizador
    optimizador.zero_grad()

    # 4. Pérdida al revés
    perdida.backward()

    # 5. Progreso del optimizador
    optimizador.step()

    ### Función de prueba

    # Pon el modelo en modo evaluación
    model_a.eval()

    with torch.inference_mode():
    
      # 1. Reenviar datos de prueba
      prueba_predc = model_a(X_prueb)

      # 2. Calcular la pérdida en datos de prueba
      prueb_perd = fn_perd(prueba_predc, y_prueb.type(torch.float))

      # Imprime lo que está pasando
      if epoca % 10 == 0:
        entrenamiento_loss.append(perdida.detach().numpy())
        test_loss.append(prueb_perd.detach().numpy())
        print(f"Epoca: {epoca} | Entrenamiento pérdida: {perdida} | Test pérdida {prueb_perd}")

# Traza las curvas de pérdida
plt.plot(entrenamiento_loss, label="Perd entrenamiento")
plt.plot(test_loss, label="Perd prueba")
plt.ylabel("Pérdida")
plt.xlabel("Epoca")
plt.legend()


# 1. Configura el modelo en modo de evaluación
model_a.eval()

# 2. Configura el administrador de contexto del modo de inferencia
with torch.inference_mode():

# 3. Asegúrate de que los cálculos se realicen con el modelo y los datos en el mismo dispositivo en nuestro caso, nuestros datos y modelo están en la CPU de forma predeterminada
  # model_1.to(device)
  # X_prueb = X_prueb.to(device)
  y_predc = model_a(X_prueb)
  
plot_predictions(datos_ent=X_ent, etiq_ent=y_ent, datos_prueba=X_prueb,  etiq_prueba=y_prueb, predictions=y_predc, name='fitted')

## https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
## https://www.kaggle.com/code/aneridalwadi/time-series-with-pytorch
## https://medium.com/@mnitin3/pytorch-forecasting-introduction-to-time-series-forecasting-706cbc48768
## https://b-nova.com/en/home/content/anomaly-detection-with-random-forest-and-pytorch