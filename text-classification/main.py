from tabnanny import check
import torch
import torchtext
from torchtext.datasets import DBpedia
print(f"torchtext version = {torchtext.__version__}")


## Data process (pipeline)
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_iter = DBpedia(split="train")
tokenizador = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, texto in data_iter:
        yield tokenizador(texto)

vocab = build_vocab_from_iterator(yield_tokens(data_iter=train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Testing vocabulary builder
print(vocab(tokenizador("Hello how are you? I am a platzi student")))

text_pipeline = lambda x: vocab(tokenizador(x))
label_pipeline = lambda x: int(x) - 1


## To process batch data
device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu") 
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (label, text) in batch:
        label_list.append(label_pipeline(x=label))
        processed_text = torch.tensor(data=text_pipeline(x=text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(dim=0))

    label_list = torch.tensor(data=label_list, dtype=torch.int64)
    offsets = torch.tensor(data=offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(tensors=text_list)

    return label_list.to(device=device), text_list.to(device=device), offsets.to(device=device)


## Building model
from classes import ModeloClasificacionTexto

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
embedding_size = 100
modelo = ModeloClasificacionTexto(vocab_size=vocab_size, embed_dim=embedding_size, num_class=num_class).to(device=device)
print(modelo)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"El modelo tiene {count_parameters(model=modelo):,} parámetros entrenables")

## ###
"""
    'Para entrenamiento'
"""

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from training import entrena
from testing import evalua


EPOCHS = 4
LEARNING_RATE = 0.2
BATCH_SIZE = 16

# Pérdida, optimizador
criterio = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=modelo.parameters(), lr=LEARNING_RATE)

# Obten el trainset y testset
train_iter, test_iter = DBpedia()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

train_dataset_size = len(train_dataset)
train_size = int(train_dataset_size * 0.95)
valid_size = train_dataset_size - train_size

# Creamos un dataset de validación con el 5% del trainset
split_train, split_valid = random_split(dataset=train_dataset, lengths=[train_size, valid_size])

# Creamos dataloaders listos para ingresar a nuestro modelo
train_dataloader = DataLoader(dataset=split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(dataset=split_valid, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

## Entrenamiento
major_loss_validation = float('inf')
for epoch in range(1, EPOCHS + 1):
    entrenamiento_acc, entrenamiento_loss, modelo = entrena(modelo=modelo, dataloader=train_dataloader, optimizer=optimizer, criterio=criterio)
    validacion_acc, validacion_loss, modelo = evalua(modelo=modelo, dataloader=valid_dataloader, criterio=criterio)
    if validacion_loss < major_loss_validation:
      major_loss_validation = validacion_loss
      best_valid_loss = validacion_loss
      torch.save(obj=modelo.state_dict(), f="mejores_guardados.pt")

stop = 1





"""
'To save model state'
"""
model_state_dict = modelo.state_dict()
optimizer_state_dict = optimizer

checkpoint = {
    "model_state_dict" : model_state_dict,
    "optimizer_state_dict" : optimizer,
    "epoch" : epoch,            # type:ignore
    "loss" : best_valid_loss    # type:ignore
    }

torch.save(obj=checkpoint, f="model_checkpoint.pth")