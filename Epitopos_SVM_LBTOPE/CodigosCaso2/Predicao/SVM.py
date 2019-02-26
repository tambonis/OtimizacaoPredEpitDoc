#!/usr/bin/env python
# coding: utf-8

# In[1]:


##############################################
#Predição considerando o caso 2
#Tiago Tambonis
#2018/2019
##############################################


# In[2]:


import pickle
import numpy as np
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import os
import subprocess


# In[3]:


## Função Suvrel.

def suvrel(X, y, gamma=2.0, norm=None, distance=False):
    """
    Return: a metric tensor for the data
    X columns representing samples and lines dimentions
    y labels
    gamma is a float
    norm:{None,\"unity\",\"t-test\"}
    distance: {False, True} if True return a tuple (weights, D)
    where D is the distanca matrix of the data
    for the geometric approach method
    """

    classes = list(set(y))
    n_classes = len(classes)
    dim = X.shape[1]

    if norm is None or norm == "unity":
        mean_cl = np.zeros((n_classes, dim))
        for i, cl in enumerate(classes):
            mean_cl[i] = np.mean(X[y == cl], axis=0)

        smeans = np.zeros(dim)
        for i, j in combinations(range(n_classes), 2):
            smeans += (mean_cl[i] - mean_cl[j]) ** 2

        if gamma != 2:
            var_cl = np.zeros((n_classes, dim))
            for cl in classes:
                var_cl[cl] = np.var(X[y == cl], axis=0)
            svar = np.sum(var_cl, axis=0)
            weights = ((gamma - 2.) * svar 
                        +  gamma /( n_classes - 1) * smeans)
        else:
            weights = smeans

        weights[weights < 0] = 0

        if norm is "unity":
            weights = weights / np.var(X, axis=0)

        if distance:
            return (weights / np.sqrt(np.sum(weights ** 2)),
                    squareform(pdist(X * np.sqrt(weights))))
        else:
            return weights / np.sqrt(np.sum(weights ** 2))

    elif norm == "t-test":
        if n_classes == 2:
            mean_cl = np.zeros((n_classes, dim))
            var_cl = np.zeros((n_classes, dim))
            for i, cl in enumerate(classes):
                mean_cl[i] = np.mean(X[y == cl], axis=0)
                var_cl[i] = np.var(X[y == cl], axis=0)

            for i, j in combinations(range(n_classes), 2):
                smeans = (mean_cl[i] - mean_cl[j]) ** 2
                #tnorm = (var_cl[i] / np.sum([y == classes[i]])
                         #+ var_cl[j] / np.sum([y == classes[j]]))

                # case with equal variance. Edited by Marcelo 21/10/13
                n1 = np.sum([y == classes[i]])
                n2 = np.sum([y == classes[j]])
                tnorm = ((n1 - 1) * var_cl[i] + (n2 - 1) * var_cl[j])                     / (n1 + n2 - 2)
            if gamma != 2:
                svar = np.sum(var_cl, axis=0)
                weights = ((gamma - 2.) * svar 
                            +  gamma /( n_classes - 1) * smeans)
            else:
                weights = smeans
            weights = weights / tnorm
            weights[weights < 0] = 0

            if distance:
                return (weights / np.sqrt(np.sum(weights ** 2)),
                        squareform(pdist(X * np.sqrt(weights))))
            else:
                return weights / np.sqrt(np.sum(weights ** 2))

        else:
            print ("error: for t-test normalization the number" +
                   " of classes must be equal 2")
            return None
    else:
        print "error: norm options are None, \"unity\" and  \"t-test\""
    return None


# In[4]:


#Carregar dados

with open('../GeracaoFeaturizacao/Dados/TabeladeDados', 'rb') as fp:
        TabelaDados = pickle.load(fp)


# In[5]:


#Divisão entre amostras e classes

X = np.array(TabelaDados.drop(['Classe'], 1))
y = np.array(TabelaDados['Classe'])


# In[6]:


results_acc = []
results_acc_parametros = []
count = 1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)
for train_index, test_index in skf.split(X, y):
    
    X_treino, X_teste = X[train_index], X[test_index]
    y_treino, y_teste = y[train_index], y[test_index]
    X_treino_efetivo = np.copy(X_treino)
    
    print("#########################################")
    print("Execucao: ", count)
    count = count + 1
    print("Shapes treino: ", X_treino.shape, y_treino.shape)
    print("Shapes teste: ", X_teste.shape, y_teste.shape)

#Normalização e featurização. Dependendo do caso analisado, alterações devem ser feitas.

    normalizar = True
    usarsuvrel = True

    if normalizar:

            scaler = StandardScaler()
            X_treino = scaler.fit(X_treino).transform(X_treino)
            #X_treino_efetivo = scaler.transform(X_treino_efetivo)

    if usarsuvrel: 

        w = suvrel(X=X_treino, y=y_treino)
        w = np.sqrt(w)

        X_treino_efetivo = w*X_treino_efetivo

        X_teste = w*X_teste

    if True: #Conversão ao libsvm.

        from sklearn.datasets import dump_svmlight_file

        dump_svmlight_file(X_treino_efetivo, y_treino, 'Dados/DadosTreinoStandarScalerlibsvm',
                           zero_based=True, multilabel=False)

        dump_svmlight_file(X_teste, y_teste, 'Dados/DadosTesteStandarScalerlibsvm',
                           zero_based=True, multilabel=False)

    os.system('python libsvm-3.23/tools/grid.py -q -v 5 Dados/DadosTreinoStandarScalerlibsvm > Dados/Treinamento.ouput')

    os.system('mv DadosTreinoStandarScalerlibsvm.* Dados')

    parametros_treino = subprocess.Popen(['tail', '-n 1', 'Dados/Treinamento.ouput'], stdout=subprocess.PIPE)
    parametros_treino = str(parametros_treino.communicate()).split()

    c = parametros_treino[0][2:4]

    gamma = parametros_treino[1]

    print("C: ", c, ", gamma: ", gamma)
    print("Comando para treinamento no libsvm utilizado:")
    print(str('./libsvm-3.23/svm-train -c ' + c + ' -g ' + gamma + ' Dados/DadosTreinoStandarScalerlibsvm'))

    os.system(str('./libsvm-3.23/svm-train -c ' + c + ' -g ' + gamma + ' Dados/DadosTreinoStandarScalerlibsvm'))

    os.system('mv DadosTreinoStandarScalerlibsvm.model Dados')

    os.system('./libsvm-3.23/svm-predict Dados/DadosTesteStandarScalerlibsvm Dados/DadosTreinoStandarScalerlibsvm.model Dados/ResultadosTeste.ouput')

    predic = np.array(pd.read_csv('Dados/ResultadosTeste.ouput', sep=" ", header=None))

    acc = accuracy_score(y_teste, predic, normalize=True)

    results_acc.append(acc)
    results_acc_parametros.append(str(c+','+gamma))


# In[7]:


print(results_acc)
print(results_acc_parametros)


# In[8]:


#Salvar resultados

with open("Dados/ResultsACC", "wb") as fp:   #Pickling
    pickle.dump(results_acc, fp)   

with open("Dados/ResultsACCParametros", "wb") as fp:   #Pickling
    pickle.dump(results_acc_parametros, fp)


# In[ ]:


#get_ipython().system(u'jupyter nbconvert --to script SVM.ipynb')


# In[ ]:


print("OK")

