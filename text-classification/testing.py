import torch

def evalua(modelo, dataloader, criterio):
    modelo.eval()
    epoch_acc = 0
    total_count = 0
    epoch_loss = 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(iterable=dataloader):
            
            # Obtenemos la la etiqueta predecida
            prediccion = modelo(text, offsets)

            # Obtenemos pérdida y accuracy
            loss = criterio(prediccion, label)
            acc = (prediccion.argmax(1) == label).sum()
            
            # Llevamos el conteo de la pérdida y el accuracy para esta epoch
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            total_count += label.size(0)

    return epoch_acc/total_count, epoch_loss/total_count, modelo