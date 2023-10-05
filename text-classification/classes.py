from torch import nn
import torch.nn.functional as F

class ModeloClasificacionTexto(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(ModeloClasificacionTexto, self).__init__()
        
        # Capa de incrustación (embedding)
        self.embedding = nn.EmbeddingBag(num_embeddings=vocab_size, embedding_dim=embed_dim)
        
        # Capa de normalización por lotes (batch normalization)
        self.bn1 = nn.BatchNorm1d(num_features=embed_dim)
        
        # Capa completamente conectada (fully connected)
        self.fc = nn.Linear(in_features=embed_dim, out_features=num_class)

    def forward(self, text, offsets):
        # Incrustar el texto (embed the text)
        embedded = self.embedding(text, offsets)
        
        # Aplicar la normalización por lotes (apply batch normalization)
        embedded_norm = self.bn1(embedded)
        
        # Aplicar la función de activación ReLU (apply the ReLU activation function)
        embedded_activated = F.relu(input=embedded_norm)
        
        # Devolver las probabilidades de clase (output the class probabilities)
        return self.fc(embedded_activated)