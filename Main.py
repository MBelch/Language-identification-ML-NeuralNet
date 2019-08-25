# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 22:38:01 2019

@author: Mehdi
e"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 19:23:54 2019
ersres
@author: Mehdi
"""
#importation des bibliotheques à utiliser:
#importing all the needed modules
import numpy as np
import mesFonctions as fct
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from joblib import dump, load
from sklearn.model_selection import cross_val_score
import tkinter as tk


#dictionnaire des langues avec leurs keys:
#dictionary of languages and their keys:

dict_langues = {1.0 : "Français", 2.0 : "Portugais" , 3.0 : "Italien" , 4.0 : "Espagnol"}


#programme appelant:
#Main program:
if __name__ == "__main__":

#fonction de la bibliotheque mesFonctions effectuant le cleaning:
#method from 'mesFonctions module for the cleaning process:
    
    textFrancais = fct.fileCleaning('txt/fr/ep-00-01-18.txt',"fracaisClean3")
    textItalien = fct.fileCleaning('txt/it/ep-00-04-11.txt',"italienClean2")
    textEspagnol = fct.fileCleaning('txt/es/ep-00-04-12.txt',"espagnolClean")
    textPortugais = fct.fileCleaning('txt/pt/ep-00-01-18.txt',"portugaisClean")
    #print(text)
    listSentenceFrancais = sent_tokenize(textFrancais)
    listSentenceItalien = sent_tokenize(textItalien)
    listSentenceEspagnol = sent_tokenize(textEspagnol)
    listSentencePortugais = sent_tokenize(textPortugais)
    
    #print(listSentence)
    

#La fonction 'listSentenceEtiquete' permet d'etiqueter les textes:
#function for labeling the each sentetence to its language:

    TFrancais = fct.listSetenceEtiquete(listSentenceFrancais,1)
    TPortugais = fct.listSetenceEtiquete(listSentencePortugais,2)
    TItalien = fct.listSetenceEtiquete(listSentenceItalien,3)
    TEspagnol = fct.listSetenceEtiquete(listSentenceEspagnol,4)
    
    
#Pour regrouper tous les phrases des textes en une seule matrice. Cette matrice par la suite sera utilisée comme data set:
#concatenate all the sentences to use it as one corpus for the training part:
    langues = np.concatenate((TFrancais,TPortugais,TItalien,TEspagnol))

    #print(langues)

#preparer la dataset/dataFrame pour extraire les n-grams en utilisant le tf-idfVectorizer:
#preparing the dataframe for the extaction of the n-gram letters, but before that we have to schuffle the sentences:
 
    languesModel=fct.melangerLangues(langues)
    df_langues = pd.DataFrame(languesModel)
    df_langues.columns = ['Phrases', 'clé_langue']
    df_langues['clé_langue'] = df_langues['clé_langue'].apply(float)
    df_langues['langue'] = df_langues['clé_langue'].map(dict_langues)


    X = df_langues['Phrases']
    y_labels = df_langues['clé_langue']

    #pour visualiser le dataFrame:
    '''print(df_langues.head(10))'''

    #on voit si on a un nombre equitable de phrases pour chaque langue
    #here we have to check that there is the same number of sentences for each language:
    '''
    clé_langue, nombrePhrase = np.unique(y_labels, return_counts=True)
    print("\n\n",dict(zip(clé_langue, nombrePhrase)))
    
    '''
#Phase d'apprentissage:
#Training set:
    
    
    #on fait le split du dataset, 0.7 pour l'entrainement et 0.3 pour le test:
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size = 0.3, random_state = 0)
    

    
    #print("La taille des features : ",len(X))
    
    #TfIdfvectorizer donne des features qui seront les n-grams du dataset: 
    #algo = MLPClassifier(solver='sgd', hidden_layer_sizes=(12,),activation = 'identity')
    algo = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,), activation = 'relu')
    vectorizer = TfidfVectorizer(analyzer='char',token_pattern = r"(?u)\b\w+\b", ngram_range=(2,3))
    
    #On utilise le Pipeline pour combiner le return du TfidVectorizer avec le input de MLPClassifier:
    classification_text = Pipeline([('tfidfVectorizer', vectorizer),('algo', algo)])

    #l'algo MLPClassifier effectue la phase de l'apprentissage
    classification_text.fit(X_train, y_train)
    
    #Pour la validation set:
    
    cv = ShuffleSplit(n_splits=4, test_size=0.1, random_state=0)
    cv_score = cross_val_score(classification_text, X_train, y_train, cv=cv)  
    print("\n\nValidation set : ",cv_score)
    
    #Sauvegarder le modele:
    
    filename = 'D:/ENSIAS_M/2ème année/ProjetIA/modele'
    '''
    dump(classification_text, filename+".joblib")
    '''
    
    #Phase de test:
    
    modele = classification_text
    #modele = load(filename+".joblib")
    


 
    prediction = modele.predict(X_test)
    

    matConf = confusion_matrix(y_test,prediction)
    #print(matConf)
    
#affichage de la matrice de confusion:
    
     
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(matConf, cmap=plt.cm.Blues, alpha=0.3)
    
    for i in range(matConf.shape[0]):
        for j in range(matConf.shape[1]):
            ax.text(x=j, y=i,s=matConf[i, j], va='center', ha='center')

    plt.xlabel('la classe predite (predict labels)')
    plt.ylabel('les vrais labels (True labels)')
    plt.show()

    
#la precision de l'algo et le rapport de classification:
#the accuracy and the classification report    

    
    
    print("\n\nLa précison de l'algorithme utilisé : ",accuracy_score(y_test,prediction))
    print("\n\n",classification_report(y_test, prediction, target_names=dict_langues.values()))
    #print(prediction)
    
    

#phase de prediction:
#predecting set:

    def predictionPhase():
        phrase = entry_field1.get()
        phrase = sent_tokenize(phrase)
        phrase = fct.listSetenceEtiquete(phrase,0)
        #print(phrase)
        df_phrase = pd.DataFrame(phrase)
        #print(df_phrase)
            
        #df_phrase.clomns = ['phrase_input', 'clé']
        #display(df_phrase)
            
        X_aPredire = df_phrase[0]
    
    
        #print(X_aPredire)
        prediction = modele.predict(X_aPredire)
        for e in prediction:
            cle = e
        return("\n\nLa phrase saisie est du : "+dict_langues[cle])
    
    def predictionDisplay():
    
        pred = predictionPhase()
    
        #pour l'affichage du text field dans l'interface:
        pred_display = tk.Text(master = window, height=5, width=50,bg='light cyan')
        
        pred_display.grid(column = 2, row = 6)
    
        pred_display.insert(tk.END, pred)
    
    
    window = tk.Tk()
    window.title("Projet Machine Learning pour detection des langues")
    window.geometry("1200x400")
    window.configure(background = "azure")
    #Label:
    label1 = tk.Label(text = "Insérez une phrase : ", font = ("Times New Roman",13), bg = "LightSkyBlue1", width = 20)
    label1.grid(column = 0, row = 2)

    #button
    bouton1 = tk.Button(text = 'Click', command = predictionDisplay, width = 11, font=('Tahoma',10))
    bouton1.grid(column = 2, row =3)

    #entry field:
    entry_field1 = tk.Entry(width = 100)
    entry_field1.grid(column=2,row=2)


    window.mainloop()







