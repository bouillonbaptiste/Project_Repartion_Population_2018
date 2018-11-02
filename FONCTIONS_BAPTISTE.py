# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:23:53 2018

@author: e1503169
"""

"""
###############################################################################
### LIBRARIES #################################################################
###############################################################################
"""
import pandas as pd
import numpy as np
import os as os
import matplotlib.pyplot as plt
import glob as glob

pd.options.mode.chained_assignment = None #default = Warn




"""

###############################################################################

#####   BAPTISTE   ############################################################

###############################################################################

Liste des fonctions:
    nbOcc
    firstOcc
    tabOcc
    rajoutAnnee
    imputOpti
    transpose_DATA

"""

#################
### FUNCTIONS ###
#################

"""
nbOcc : fonction qui renvoie le nombre d'éléments existants dans un tableau
@params :
    tableau : un tableau n * 1
@return :
    tt : le nombre d'éléments
"""
def nbOcc(tableau):
    tt=0
    for i in range(len(tableau)):
        if not(pd.isnull(tableau.iloc[i])):
            tt=tt+1
    return tt


"""
firstOcc : fonction qui renvoie le premier élément dans un tableau
@params :
    tableau : un tableau n * 1
@return :
    i : l'index du premier élément
"""
def firstOcc(tableau):
    i=0
    while pd.isnull(tableau.iloc[i]):
        i=i+1
    return i


"""
tabOcc : fonction qui renvoie l'index du i ème élément dans un tableau
@params :
    tableau : un tableau n * 1
    i : la i ème valeur du tableau
@return :
    k : l'index du i ème élément
"""
def tabOcc(tableau,i):
    ttmax=nbOcc(tableau) #le nombre d'occurences du tableau
    if i>ttmax or i<0: # on rejette si jamais on demande une occurence supérieure au nombre max du tableau
        return None
    else: # sinon on regarde à quel index se trouve la i ème occurence
        j=0
        tt=0
        while tt<i:
            j=j+firstOcc(tableau.iloc[j:]) # l'index vaut lui même plus le nombre de pas entre lui et la prochaine occurence
            j=j+1
            tt=tt+1 # on compte le nombre d'occurences passées, pour savoir quand s'arrêter
        return (j-1) # j-1 car à la dernière étape on a ajouté 1 à j



"""
Maintenant il faut une fonction qui optimise l'imputation
"""

"""
rajoutAnnee : fonction qui créé un tableau de la période désirée
@params :
    tableau : le tableau des observations
    annee_debut : l'année de début
    annee_fin : l'année de fin
@return :
    tabajout : le tableau de la période désirée
"""
def rajoutAnnee(tableau,annee_debut,annee_fin,dep,variable):
    
    List_annes = np.unique(tableau["Année"])
    
    DEP = list(np.repeat(dep,annee_fin-annee_debut+1,axis=0))
    ANNEE = list(range(annee_debut,annee_fin+1))
    DATA = list(np.repeat(0,annee_fin-annee_debut+1,axis=0))
    
    tabajout = pd.DataFrame({"Lieu":DEP,
                             "Année":ANNEE,
                             variable:DATA})
    # ii = 0
    # ii += 1
    for ii in range(len(tabajout)):
        if (tabajout["Année"].iloc[ii] in List_annes) and list(~pd.isnull(tableau[tableau["Année"]==tabajout["Année"].iloc[ii]][variable]))[0]:
            tabajout[variable].iloc[ii] = float(tableau[tableau["Année"]==tabajout["Année"].iloc[ii]][variable])
        else:
            tabajout[variable].iloc[ii] = None
    
    """
    # ajout au début
    while(tabajout["Année"].iloc[0]>annee_debut):
        tmp = pd.DataFrame([tabajout["Lieu"].iloc[0],tabajout["Année"].iloc[0]-1,None]).transpose()
        tmp.columns = tabajout.columns
        tabajout = tmp.append(tabajout)
    
    # ajout à la fin
    while(tabajout["Année"].iloc[-1]<annee_fin):
        tmp = pd.DataFrame([tabajout["Lieu"].iloc[0],tabajout["Année"].iloc[-1]+1,None]).transpose()
        tmp.columns = tabajout.columns
        tabajout = tabajout.append(tmp)
    """
    
    return tabajout
    

"""
imputOpti : fonction qui optimise l'imputation de valeurs dans un tableau
@params :
    tableau : un tableau n * 1 avec 2 éléments ou plus
@return :
    tabbis : un tableau n * 1 imputé
"""
def imputOpti(tableau,annee_debut,annee_fin):
    
    ttmax=nbOcc(tableau) #le nombre d'éléments du tableau
    
    
    tabbis=tableau #on copie le tableau pour ne pas modifier le paramètre d'entrée
    """
    étape 1 : consiste à imputer des valeurs pour les années antérieures à la première occurence
    """
    aa=tabOcc(tabbis,1)
    bb=tabOcc(tabbis,2)
    cc=(tabbis.iloc[bb]-tabbis.iloc[aa])/(bb-aa) # le coef directeur de la droite
    dd=tabbis.iloc[aa]-((annee_debut+aa)*cc) # l'ordonnée à l'origine
    for i in range(aa): # pour toutes les années avant la première occurence
        if pd.isnull(tabbis.iloc[i]): # si il n'y a pas de valeur renseignée
            tabbis.iloc[i]=dd+(annee_debut+i)*cc # alors on impute avec une estimation linéaire
    # fin étape 1
    
    """
    étape 2 : imputer des valeurs entre deux occurences de facon linéaire
    """
    tt=aa # la première occurence
    while tt<ttmax: #tant que l'élément courant n'est pas le dernier
        ttmax=nbOcc(tabbis) #on met à jour ttmax car comme on impute, on rajoute des occurences
        
        aa=tabOcc(tabbis,tt)
        bb=tabOcc(tabbis,tt+1)
        cc=(tabbis.iloc[bb]-tabbis.iloc[aa])/(bb-aa) # le coef directeur de la droite
        dd=tabbis.iloc[aa]-((annee_debut+aa)*cc) # l'ordonnée à l'origine
        for i in range(aa,bb): # pour chaque année entre deux occurences
            if pd.isnull(tabbis.iloc[i]): # si aucune valeur n'est renseignée
                tabbis.iloc[i]=dd+(annee_debut+i)*cc # alors on l'estime linéairement
        tt=tt+1 # on a une nouvelle occurence dans le tableau
    # fin étape 2
    
    """
    étape 3 : consiste à imputer les années après la dernière occurence pour aller jusqu'à l'année maximale
        ex : si les valeurs sont renseignées jusqu'à 2012, on va aller jusqu'à 2014
    """
    aa=tabOcc(tabbis,ttmax-1)
    bb=tabOcc(tabbis,ttmax)
    cc=(tabbis.iloc[bb]-tabbis.iloc[aa])/(bb-aa) # le coef directeur de la droite
    dd=tabbis.iloc[aa]-((annee_debut+aa)*cc) # l'ordonnée à l'origine
    """
    # ANCIEN
    for gg in range(bb,len(tableau)): # pour toutes les années après la dernière occurence
        if pd.isnull(tabbis.iloc[gg]): # si aucune valeur n'est renseignée
            tabbis.iloc[gg]=dd+(annee_debut+i)*cc # on estime linéairement
    # END
    """
    # NOUVEAU
    for gg in range(bb,annee_fin-annee_debut+1): # pour toutes les années après la dernière occurence
        if pd.isnull(tabbis.iloc[gg]): # si aucune valeur n'est renseignée
            tabbis.iloc[gg]=dd+(annee_debut+gg)*cc # on estime linéairement
    # END
    # fin étape 3
    
    for ii in range(len(tabbis)):
        if tabbis.iloc[ii]<0:
            tabbis.iloc[ii] = 0
    
    return tabbis

"""
recreerDep : fonction qui reconstruit un bon jeu de données pour une variable pour une département
@params :
    DATA : une liste ou df
    annee_debut : le début des observations
    annee_fin : la fin des observations
    variable : le nom de la variable étudiée
@return :
    retour : un df contenant tout ce dont nous avons besoin
"""
def recreerDep(DATA,dep,annee_debut,annee_fin,variable):
    TMP = list(DATA)
    DEP = list(np.repeat(dep,annee_fin-annee_debut+1,axis=0))
    #len(DEP)
    #len(ANNEE)
    ANNEE = list(range(annee_debut,annee_fin+1))
    
    
    retour = pd.DataFrame({"Lieu":DEP,
                           "Année":ANNEE,
                           variable:TMP})
    
    return retour

"""
transpose_DATA : fonction qui transpose les données d'entrée de manière intélligente
@params :
    pays : le pays des données transposées
    DATA : le jeu de données complet
@return :
    retour : les données transposées avec le pays gardé bien comme il faut
"""
def transpose_DATA(pays,DATA):
    ret_data = DATA.transpose()
    ret_pays = pd.DataFrame({'Pays':[pays]*len(ret_data)},index=ret_data.index)
    retour = pd.concat([ret_pays,ret_data],ignore_index=True,axis=1)
    
    new_col = retour.iloc[0,:]
    new_col.is_copy = False
    
    retour = retour.iloc[1:,:]
    new_col[0] = "Lieu"
    retour.columns = new_col
    
    retour = retour.reset_index()
    # retour = retour.set_index(["index","Lieu"])
    
    return retour









