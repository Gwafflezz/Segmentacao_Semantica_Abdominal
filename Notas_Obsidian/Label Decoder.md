---
tags: 
Título: Conversor de Matriz de inteiros para Matriz RGB 
Local: Pc
Data: 14-11-2023
Subprojeto: Pŕe processamento
Participantes: Davi
---
Projeto: [[Segmentação_Semântica_Abdominal]]
Referências:  [[Label Encoder]]

---
Recebe uma matriz 2D de inteiros onde cada pixel corresponde a um rótulo do dicionário. Converte esse array de volta para uma imagem RGB

---
A saída de um modelo de segmentação geralmente é da forma [batch_size, n_classes, altura, largura]. n_classes é um *one hot vector*, para um dado índice de pixel, com tamanho n-classes. A função softmax define uma probabilidade para cada classe, e a maior é tida como a cor correspondente ao pixel.
