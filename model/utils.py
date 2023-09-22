import matplotlib.pyplot as plt

# matplotlib.use("TkAgg")

def plot_predictions(datos_ent, etiq_ent, datos_prueba, etiq_prueba, predictions, name):
  """
  Traza datos de entrenamiento, datos de prueba y compara predicciones
  """
  plt.figure(figsize=(10, 10))
  # Traza datos de entrenamiento en verde
  plt.scatter(x=datos_ent, y=etiq_ent, c="g", s=6, label="Datos de entrenamiento") 
  # Traza datos de prueba en amarillo
  plt.scatter(x=datos_prueba, y=etiq_prueba, c="y", s=6, label="Datos de prueba")
  if predictions is not None:
    # Traza las predicciones en rojo
    plt.scatter(x=datos_prueba, y=predictions, c="r", s=6, label="Predicciones")
  # Leyenda
  plt.legend(prop={"size": 12})
  plt.savefig(f'{name}.png')