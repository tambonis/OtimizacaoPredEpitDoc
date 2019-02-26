#!/usr/bin/env python
# coding: utf-8

# In[1]:


##############################################
#Geração de dados.
#Tiago Tambonis
#2018/2019
##############################################


# In[2]:


from Bio import SeqIO #Para leitura das sequências
import numpy as np
from pydpi.pypro import PyPro
from pydpi.pypro import CTD
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

protein = PyPro() #Global
kmaxlag = 19


# In[3]:


#Geração das características

def getcaracteristicas(seqs, label): 
    
    #Composição de Dipeptídeos
    def getcaracteristicasDPComp(pep, kseq):

        protein.ReadProteinSequence(pep)

        kDPComp = protein.GetDPComp()
        kDPComp = pd.DataFrame(kDPComp.items(), columns=['PepAAComp', kseq])
        kDPComp = kDPComp.set_index('PepAAComp')

        return(kDPComp)

    for i in range(len(seqs)):

        if i==0: DPCompData = getcaracteristicasDPComp(seqs[i], kseq=str(label+str(i+1)))
        else: DPCompData = DPCompData.merge(getcaracteristicasDPComp(seqs[i], kseq=str(label+str(i+1))), 
                                left_on='PepAAComp', right_on='PepAAComp', how='inner')
                
    Dados = [DPCompData] #Quando for colocar outros descritores é só adicionar aqui
    Dados = pd.concat(Dados)
    Dados = Dados.T #Transpor para adequar aos pacotes
    
    return(Dados)


# # Leitura das sequências positivas

# In[4]:


#Leitura das sequências FASTA

seqspositivas = []
for record in SeqIO.parse("Dados/LBtope_Fixed_non_redundant_Positive_pattern.txt.fasta", "fasta"):
    #print(record.seq)
    seqspositivas.append(record.seq)
#seqspositivas = seqspositivas[0:50]


# In[5]:


TabelaDadosPositivas = getcaracteristicas(seqs=seqspositivas, label="SeqPos")


# In[6]:


TabelaDadosPositivas['Classe'] = np.repeat(1, TabelaDadosPositivas.shape[0]).tolist()


# # Leitura das sequências negativas

# In[7]:


#Leitura das sequências FASTA

seqsnegativas = []
for record in SeqIO.parse("Dados/LBtope_Fixed_non_redundant_Negative_pattern.txt.fasta", "fasta"):
    #print(record.seq)
    seqsnegativas.append(record.seq)
#seqsnegativas = seqsnegativas[0:50]


# In[8]:


TabelaDadosNegativas = getcaracteristicas(seqs=seqsnegativas, label="SeqNeg")


# In[9]:


TabelaDadosNegativas['Classe'] = np.repeat(-1, TabelaDadosNegativas.shape[0]).tolist()


# # Binding dados

# In[10]:


TabelaDadosPositivas.head()


# In[11]:


TabelaDadosNegativas.tail()


# In[12]:


TabelaDados = [TabelaDadosPositivas, TabelaDadosNegativas]
TabelaDados = pd.concat(TabelaDados)


# In[13]:


TabelaDados.head()


# In[14]:


TabelaDadosPositivas.head()


# In[15]:


TabelaDados.tail()


# In[16]:


TabelaDadosNegativas.tail()


# In[17]:


#Salvar tabela de dados principal 

with open("Dados/TabeladeDados", "wb") as fp:   #Pickling
    pickle.dump(TabelaDados, fp)   


# In[18]:


print("OK")

