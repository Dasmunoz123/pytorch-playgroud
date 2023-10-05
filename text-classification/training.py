import torch


def entrena(modelo, dataloader, optimizer, criterio):
    # Colocar el modelo en formato de entrenamiento
    modelo.train()

    # Inicializa accuracy, count y loss para cada epoch
    epoch_acc = 0
    epoch_loss = 0
    total_count = 0 

    for idx, (label, text, offsets) in enumerate(iterable=dataloader):
        # reestablece los gradientes después de cada batch
        optimizer.zero_grad()
        # Obten predicciones del modelo
        prediccion = modelo(text, offsets)

        # Obten la pérdida
        loss = criterio(prediccion, label)
        
        # backpropage la pérdida y calcular los gradientes
        loss.backward()
        
        # Obten la accuracy
        acc = (prediccion.argmax(1) == label).sum()
        
        # Evita que los gradientes sean demasiado grandes 
        torch.nn.utils.clip_grad_norm_(parameters=modelo.parameters(), max_norm=0.1)   # type: ignore

        # Actualiza los pesos
        optimizer.step()

        # Llevamos el conteo de la pérdida y el accuracy para esta epoch
        epoch_acc += acc.item()
        epoch_loss += loss.item()
        total_count += label.size(0)

        if idx % 500 == 0 and idx > 0:
          print(f" epoca | {idx}/{len(dataloader)} batches | perdida {epoch_loss/total_count} | accuracy {epoch_acc/total_count}")

    return epoch_acc/total_count, epoch_loss/total_count, modelo