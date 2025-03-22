---
tags: 
Título: Label_Encoder
Local: Pc
Data: 14-11-2024
Projeto: [[Segmentação_Semântica_Abdominal]]
Participantes: Davi
---
---
Recebe um array numpy de uma imagem RGB e converte para uma matriz 2D de inteiros, onde cada pixel é um inteiro de uma classe definida no dicionário.

---
- A matriz 2D de inteiros pode ser usada como eixo Y em uma função de perda
- Utilizar operações matriciais para reduzir complexidade

**np.all:** cria uma máscara booleana e faz a comparação  [RGB] x dict.
havendo correpondência, o rótulo é adicionado no pixel de mesmo índice na matriz de zeros.



---
Referências externas:https://medium.com/@noah.vandal/creating-a-label-class-matrix-from-an-rgb-mask-for-segmentation-training-in-python-2ddceba459cb
    