---
tags: 
Título: Log de desenvolvimento
Local: Casa
Data: 10-11-2024
Projeto: ProjetoFinal_LAMIA
Participantes: Davi, Rhayron
---
Projeto:
Referências:  

# Síntese
---
Nota onde vou escrever as atividades diárias relacionadas ao projeto: tarefas realizadas, dúvidas, problemas, etc

---
#### 10/11 

- criação da lista com recursos e datasets disponíveis
- Leitura do artigo original U-net
- Leitura de implementações diversas, com pytorch e tensorflow

falta escolher a tarefa específica da rede,  para poder baixar o dataset e começar a implementar.
- dataset: https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm/data

#### 11/11
- Início da implementação com tensorflow-keras
- Dataset ta faltando anotações para as imagens de treinamento em US reais

#### 12/11
- Implementação da U-net 
- Início do dataloader
#### 13/11
- [[Dataloader]]: - img_loader, carregamento dos datasets, dicionário RGB e função de label encoder
#### 14/11
criadas as funções: 
- gen_dict_core(): dicionário de classes de cor
- Conversor Matriz RGB > Matriz de classes
- Conversor Matriz One-Hot (h, w, 9) > RGB (h,w,3)
Atualização do fluxograma

#### 15/11
- [x] Teste de rgb>one-hot ✅ 2024-11-15
- [ ] Popular datasets de treino e validação definitivos
		Não sei se devo unificar os diretórios de treino(633) e teste(293).
		train_test_split noa ja faz isso? 
		- se eu concatenar as listas, elas continuarão com a mesma ordem?
		- Só passo uma lista já populada pelo dataloader?
- [ ]  Codificar máscaras rgb>one-hot
- 

  