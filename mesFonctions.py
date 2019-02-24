# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:54:41 2019

@author: Mehdi
"""
import re
import string
import numpy as np

def fileCleaning(fileDirectory,langueNumero):
    
    """fonction effectuant le cleaning des fichiers textes"""
    
    #dans ce bout de code, on ouvre le fichier en mode ecriture et on elimine les balises dans le texte
    file = open(fileDirectory,'rt',encoding = 'utf-8')  
    text = file.read()
    file.close()
    text = re.sub("<.*?>","",text)
    
    #dans ce bout de code, on elimine la ponctuation:
    for ch in string.punctuation:
        if ch != "'" and ch !=".":                                                                                                     
            text = text.replace(ch, " ")
            
        
    #en elimine les nombre du text:
    for e in string.digits:
        text = text.replace(e," ")
        
   
        
    #ecrire le text nettoyé dans la 'Directory' mentionnée
    filename = 'D:/ENSIAS_M/2ème année/ProjetIA/texte netoye/'+langueNumero+'.txt'
    file = open(filename , 'wt') 
    for e in text:
        file.write(e)
        

    
    file.close()
    
    file = open(filename , 'rt')
    text = file.read()
    file.close()
    
   
    
    return text

def ouvrirTextNetoye(path):
    file = open(path,'rt',encoding = 'utf-8')
    text = file.read()
    file.close
    return text


#fonction pour attribuer a chaque phrases l'indice de la langue:
def listSetenceEtiquete(listSentence, indiceLangue):
    
    n = len(listSentence)
    listSentence = np.array(listSentence)
    listSentence = listSentence.reshape(n,1)
    listIndex = np.zeros((n,1))
    
    for i in range(len(listIndex)):
        listIndex[i] = listIndex[i] + indiceLangue
    
    listLangueEtiquete = np.hstack((listSentence, listIndex))
    return listLangueEtiquete

#fonction pour melanger les phrases dans notre matrice:
def melangerLangues(langues):
    np.random.shuffle(langues)
    return langues

#fonction des grams:
'''
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 3),token_pattern = r"(?u)\b\w+\b",analyzer='char')

corpus = ['Mehdi','Belchiti','embi','ensias','Benbrahim']

X = vectorizer.fit_transform(corpus)
analyze = vectorizer.build_analyzer()
#print(vectorizer.get_feature_names())
print(vectorizer.transform(['benbrahim']).toarray())
''' 
    
#programme appelant:
'''
text1 = fileCleaning('D:/ENSIAS_M/2ème année/ProjetIA/txt/fr/ep-00-01-18.txt',"francais2")
print(text1)
'''