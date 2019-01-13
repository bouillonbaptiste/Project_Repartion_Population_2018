# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:45:29 2018

@author: Baptiste
"""


"""
###############################################################################
### LIBRARIES #################################################################
###############################################################################
"""

import pandas as pd
import numpy as np

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster 

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import statsmodels.api as sm

import matplotlib.pyplot as plt

import os as os

curr_dir = "C:/Users/Baptiste/Desktop/projet_lardjane/RENDU/"
os.chdir(curr_dir)




"""
###############################################################################
### DICTIONNARIES #############################################################
###############################################################################
"""

variables_expl = {"Population_Feminine_0_19"    : 0,
                  "Population_Feminine_19_39"   : 1,
                  "Population_Feminine_40_59"   : 2,
                  "Population_Feminine_60_74"   : 3,
                  "Population_Feminine_75"      : 4,
                  "Population_Francaise_Totale" : 5,
                  "Population_Masculine_0_19"   : 6,
                  "Population_Masculine_19_39"  : 7,
                  "Population_Masculine_40_59"  : 8,
                  "Population_Masculine_60_74"  : 9,
                  "Population_Masculine_75"     : 10}

variables_expl_3 = {"Population_Feminine_0_19"  : 0,
                  "Population_Feminine_19_39"   : 1,
                  "Population_Feminine_40_59"   : 2,
                  "Population_Feminine_60_74"   : 3,
                  "Population_Feminine_75"      : 4,
                  "Population_Masculine_0_19"   : 5,
                  "Population_Masculine_19_39"  : 6,
                  "Population_Masculine_40_59"  : 7,
                  "Population_Masculine_60_74"  : 8,
                  "Population_Masculine_75"     : 9}

"""
###############################################################################
### IMPORTATIONS ##############################################################
###############################################################################
"""

PREDICTEUR = pd.read_csv("DATA_ALL2.csv",sep=";",header=0)
PREDICTRICE = pd.read_csv("PREDIRE.csv",sep=";",header=0)


PREDICTEUR2 = PREDICTEUR.drop(["Annee","Lieu"],1)
PREDICTRICE2 = PREDICTRICE.drop(["Annee","Lieu"],1)


index1995 = PREDICTEUR[PREDICTEUR["Annee"]==1995].index
PREDICTEUR_ACP1 = PREDICTEUR[PREDICTEUR["Annee"]==1995].drop(["Annee","Lieu"],1)
PREDICTRICE_ACP1 = PREDICTRICE[PREDICTRICE["Annee"]==1995].drop(["Annee","Lieu"],1)

index2005 = PREDICTEUR[PREDICTEUR["Annee"]==2005].index
PREDICTEUR_ACP2 = PREDICTEUR[PREDICTEUR["Annee"]==2005].drop(["Annee","Lieu"],1)
PREDICTRICE_ACP2 = PREDICTRICE[PREDICTRICE["Annee"]==2005].drop(["Annee","Lieu"],1)

index2015 = PREDICTEUR[PREDICTEUR["Annee"]==2015].index
PREDICTEUR_ACP3 = PREDICTEUR[PREDICTEUR["Annee"]==2015].drop(["Annee","Lieu"],1)
PREDICTRICE_ACP3 = PREDICTRICE[PREDICTRICE["Annee"]==2015].drop(["Annee","Lieu"],1)

"""
###############################################################################
### PREDICTRICE3 ############################################################
###############################################################################
"""


PREDICTRICE3 = PREDICTRICE2[["Population_Feminine_0_19",
                                "Population_Feminine_19_39",
                                "Population_Feminine_40_59",
                                "Population_Feminine_60_74",
                                "Population_Feminine_75",
                                "Population_Masculine_0_19",
                                "Population_Masculine_19_39",
                                "Population_Masculine_40_59",
                                "Population_Masculine_60_74",
                                "Population_Masculine_75"]].multiply(1./PREDICTRICE2["Population_Francaise_Totale"],axis=0)



"""
###############################################################################
### FONCTIONS BAPTISTE ########################################################
###############################################################################
"""

"""
Summary_lm : fonction qui retourne l'équivalent d'un summary d'une lm de R sous python
@params:
    X : un dataframe contenant les valeurs des prédicteurs
    Y : un dataframe contenant les variables réponse
@return :
    myDF3 : un dataframe contenant quatre colonnes, les coefficients, les ecart-types, la valeurs du quantile de student associé, la p-value
"""
def Summary_lm(X,Y):
    lm = LinearRegression()
    lm.fit(X,Y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((Y-predictions)**2))/(len(newX)-len(newX.columns))
    
    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    """
    sd_b = np.round(sd_b,
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    """
    myDF3 = pd.DataFrame()
    myDF3["coefficients"],myDF3["se"],myDF3["t_values"],myDF3["p_values"] = [params,sd_b,ts_b,p_values]
    
    myDF3.index = ["Intercept"]+list(X)
    
    return myDF3

"""
Summary_log : fonction qui retourne l'équivalent d'un summary d'une glm(family=binomial(link='logit')) de R sous python
@params:
    X : un dataframe contenant les valeurs des prédicteurs
    Y : un dataframe contenant les variables réponse
@return :
    myDF3 : un dataframe contenant quatre colonnes, les coefficients, les ecart-types, la valeurs du quantile gaussien associé, la p-value
"""
def Summary_log(X,Y):
    # X = XX.iloc[:,0:10]
    # Y = YY
    
    modelk = sm.Logit(Y,X)
    modelk_fit = modelk.fit()
    
    aa = modelk_fit.params
    bb = modelk_fit.bse
    cc = aa/bb
    dd = modelk_fit.pvalues
    
    # modelk_fit.summary()
    
    myDF3 = pd.DataFrame()
    myDF3["coefficients"],myDF3["se"],myDF3["z_values"],myDF3["p_values"] = [aa,bb,cc,dd]
    
    return myDF3
    
    
    


"""
# Recup.pire.coef : fonction qui rend le coeficient avec la p-value la plus petite parmis un ensemble de coeficients d'une régression linéaire multiple
# @ parmas : 
#   data.summary : un summary d'une lm
# @return :
#   return : le pire coeficient de la régression
"""
def Recup_pire_coef(data_summary):
    # on supprime tous les coefficients avec une p-value supérieure à 0.05
    sum_model = data_summary[data_summary["p_values"]>0.05]
    # et on récupère le nom des coefficients
    retour = sum_model.index
    if(len(retour)==0): # si tous les coefficients ont une p-value inférieure à 0.05 ...
        # on récupère le coefficient avec la p-value la plus grande
        sum_model = data_summary.sort_values(by="p_values",axis=0)
        retour = sum_model.index[-1]
    return(retour)


"""
# Recup.pire.Recup_pire_coef_log : fonction qui rend le coeficient avec la p-value la plus petite parmis un ensemble de coeficients d'une regression logistique
# @ parmas : 
#   data.summary : un summary d'une log
# @return :
#   return : le pire coeficient
"""
def Recup_pire_coef_log(data_summary):
    ## on supprime tous les coefficients avec une p-value supérieure à 0.05
    # sum_model = data_summary[data_summary["p_values"]>0.05]
    ## et on récupère le nom des coefficients
    # retour = sum_model.index
    ##if(len(retour)==0): # si tous les coefficients ont une p-value inférieure à 0.05 ...
        ## on récupère le coefficient avec la p-value la plus grande
    sum_model = data_summary.sort_values(by="p_values",axis=0)
    retour = sum_model.index[-1]
    return(retour)

"""
K_var_explicatives_lm : fonction qui retourne les K variables explicatives d'une régression linéaire
@params :
    KK : le nombre de coefficients à retourner
    maximum : la valeur maximale au dessus de laquelle une p-value n'est plus significative
    table_PREDICTEUR : les valeurs des prédicteurs
    table_PREDICTRICE : les variables réponse
"""
def K_var_explicatives_lm(KK,maximum,table_PREDICTEUR,table_PREDICTRICE):
    variables_explicatives = list()
    # evo.R2 <- list()
    list_var = list(PREDICTRICE2)
    for variable in list_var:
          
        # variable = "Population_Feminine_0_19"
        # RR <- c()
        # VV <- c()
        
        """
        on veut sélectionner les variables le splus significative spour distinguer les individus entre eux.
        Les plus significatives au sens de la population totale
        """
        XX = table_PREDICTEUR
        YY = table_PREDICTRICE[variable] 
        n = list(PREDICTEUR2)
        try:
            n = n.remove("Intercept")
        except:
            pass
        
        model1 = Summary_lm(XX,YY)
        sum_modelk = model1.sort_values(by="p_values",axis=0)
        
        while((sum_modelk.p_values[-1] > maximum and len(sum_modelk) > 4) or (len(sum_modelk) > KK)):
          pire = Recup_pire_coef(sum_modelk)
          try:
              n = [x for x in n if x not in pire]
          except:
              pass
          
          # len(n)
          # len(pire)
          XX = PREDICTEUR2[n]
          YY = PREDICTRICE2[variable]
          
          sum_modelk =  Summary_lm(XX,YY)[1:].sort_values(by="p_values",axis=0)
          
        n = sum_modelk.index
        variables_explicatives.append(n)
        
    return variables_explicatives


"""

###############################################################################

### MODELES SUR EFFECTIFS #####################################################

###############################################################################

"""


"""
###############################################################################
### CAH #######################################################################
###############################################################################
"""


#CAH 1995 

ALL1995DON = PREDICTEUR[PREDICTEUR["Annee"]==1995].drop("Annee",1)
lieu =  ALL1995DON.Lieu
ALL1995REP = PREDICTRICE[PREDICTRICE["Annee"]==1995].drop(["Annee"],1)

ALL1995DON = ALL1995DON.set_index("Lieu")
ALL1995REP = ALL1995REP.set_index("Lieu")

ALL1995DON_norm = scale(ALL1995REP)
#(ALL1995DON - ALL1995DON.min()) / (ALL1995DON.max() - ALL1995DON.min())
ALL1995REP_norm = scale(ALL1995REP)
ALL1995DON_norm = pd.DataFrame(ALL1995DON_norm)
ALL1995REP_norm = pd.DataFrame(ALL1995REP_norm)

Z = linkage(ALL1995DON_norm,method='ward',metric='euclidean')

inertie = np.sort(Z[:,2]) # attention, la colonne change aléatoirement de place
inertie = np.flip(inertie[0:93],0)
fig = plt.figure(1, figsize=(8, 8))
plt.title ("Inertie résiduelle en fonction du nombre de classes en 1995")
plt.plot(inertie)
ptr, = plt.plot(4,inertie[4],"r.")
ptg, = plt.plot(9,inertie[9],"g.")
plt.legend([ptg,ptr],['11 classes','6 classes'])
plt.savefig('InertieDON1995.png')
plt.show()



fig = plt.figure(1, figsize=(8, 8))
plt.title ("Classification Ascendante Hiérarchique des départements en 1995")
dendrogram(Z,labels = ALL1995DON.index, orientation = 'left', color_threshold = inertie[4])
plt.savefig('CAHDON1995.png')
plt.show()
"""
groupes_cah = fcluster(Z, t=7, criterion = 'maxclust')
print(groupes_cah)

idg = np.argsort(groupes_cah)
print(pd.DataFrame(ALL1995DON.index[idg],groupes_cah[idg]))
"""

#CAH 2005

ALL1995DON = PREDICTEUR[PREDICTEUR["Annee"]==2005].drop("Annee",1)
lieu =  ALL1995DON.Lieu
ALL1995REP = PREDICTRICE[PREDICTRICE["Annee"]==2005].drop(["Annee"],1)

ALL1995DON = ALL1995DON.set_index("Lieu")
ALL1995REP = ALL1995REP.set_index("Lieu")

ALL1995DON_norm = scale(ALL1995REP)
#(ALL1995DON - ALL1995DON.min()) / (ALL1995DON.max() - ALL1995DON.min())
ALL1995REP_norm = scale(ALL1995REP)
ALL1995DON_norm = pd.DataFrame(ALL1995DON_norm)
ALL1995REP_norm = pd.DataFrame(ALL1995REP_norm)

Z = linkage(ALL1995DON_norm,method='ward',metric='euclidean')

inertie = np.sort(Z[:,2]) # attention, la colonne change aléatoirement de place
inertie = np.flip(inertie[0:93],0)
fig = plt.figure(1, figsize=(8, 8))
plt.title ("Inertie résiduelle en fonction du nombre de classes en 2005")
plt.plot(inertie)
ptr, = plt.plot(4,inertie[4],"r.")
ptg, = plt.plot(9,inertie[9],"g.")
plt.legend([ptg,ptr],['11 classes','6 classes'])
plt.savefig('InertieDON2005.png')
plt.show()



fig = plt.figure(1, figsize=(8, 8))
plt.title ("Classification Ascendante Hiérarchique des départements en 2005")
dendrogram(Z,labels = ALL1995DON.index, orientation = 'left', color_threshold = inertie[4])
plt.savefig('CAHDON2005.png')
plt.show()

groupes_cah = fcluster(Z, t=7, criterion = 'maxclust')
print(groupes_cah)

idg = np.argsort(groupes_cah)
print(pd.DataFrame(ALL1995DON.index[idg],groupes_cah[idg]))


#CAH 2015

ALL1995DON = PREDICTEUR[PREDICTEUR["Annee"]==2015].drop("Annee",1)
lieu =  ALL1995DON.Lieu
ALL1995REP = PREDICTRICE[PREDICTRICE["Annee"]==2015].drop(["Annee"],1)

ALL1995DON = ALL1995DON.set_index("Lieu")
ALL1995REP = ALL1995REP.set_index("Lieu")

ALL1995DON_norm = scale(ALL1995REP)
#(ALL1995DON - ALL1995DON.min()) / (ALL1995DON.max() - ALL1995DON.min())
ALL1995REP_norm = scale(ALL1995REP)
ALL1995DON_norm = pd.DataFrame(ALL1995DON_norm)
ALL1995REP_norm = pd.DataFrame(ALL1995REP_norm)

Z = linkage(ALL1995DON_norm,method='ward',metric='euclidean')

inertie = np.sort(Z[:,2]) # attention, la colonne change aléatoirement de place
inertie = np.flip(inertie[0:93],0)
fig = plt.figure(1, figsize=(8, 8))
plt.title ("Inertie résiduelle en fonction du nombre de classes en 2015")
plt.plot(inertie)
ptr, = plt.plot(4,inertie[4],"r.")
ptg, = plt.plot(9,inertie[9],"g.")
plt.legend([ptg,ptr],['11 classes','6 classes'])
plt.savefig('InertieDON2015.png')
plt.show()



fig = plt.figure(1, figsize=(8, 8))
plt.title ("Classification Ascendante Hiérarchique des départements en 2015")
dendrogram(Z,labels = ALL1995DON.index, orientation = 'left', color_threshold = inertie[4])
plt.savefig('CAHDON2015.png')
plt.show()

groupes_cah = fcluster(Z, t=7, criterion = 'maxclust')
print(groupes_cah)

idg = np.argsort(groupes_cah)
print(pd.DataFrame(ALL1995DON.index[idg],groupes_cah[idg]))

"""
###############################################################################
### VARIABLES EXPLICATIVES ####################################################
###############################################################################
"""

variables_explicatives = list()
# evo.R2 <- list()
list_var = list(PREDICTRICE2)
for variable in list_var:
      
    # variable = "Population_Feminine_0_19"
    # RR <- c()
    # VV <- c()
    
    """
    on veut sélectionner les variables le splus significative spour distinguer les individus entre eux.
    Les plus significatives au sens de la population totale
    """
    XX = PREDICTEUR2
    YY = PREDICTRICE2[variable] 
    n = list(PREDICTEUR2)
    try:
        n = n.remove("Intercept")
    except:
        pass
    
    model1 = Summary_lm(XX,YY)
    sum_modelk = model1.sort_values(by="p_values",axis=0)
    
    while((sum_modelk.p_values[-1] > 1e-50 and len(sum_modelk) > 4) or (len(sum_modelk) > 10)):
      pire = Recup_pire_coef(sum_modelk)
      try:
          n = [x for x in n if x not in pire]
      except:
          pass
      
      # len(n)
      # len(pire)
      XX = PREDICTEUR2[n]
      YY = PREDICTRICE2[variable]
      
      sum_modelk =  Summary_lm(XX,YY)[1:].sort_values(by="p_values",axis=0)
      
    n = sum_modelk.index
    variables_explicatives.append(n)

for variable in list_var:
    ii = variables_expl[variable]
    print(variable)
    df_tmp = pd.DataFrame(variables_explicatives[ii])
    df_tmp.columns = ['variables']
    print(df_tmp)

"""
variable = 'Population_Masculine_75'
ii = variables_expl[variable]
df_tmp = pd.DataFrame(variables_explicatives[ii])
df_tmp.columns = ['variables ' + variable]
print(df_tmp)
"""

"""
###############################################################################
### BOOTSTRAP1 ################################################################
###############################################################################
"""


R2_hat = list()
REPET = 1500

for tranche_age in range(len(variables_explicatives)):
    # tranche_age = 0
    population = list_var[tranche_age]
    var_pop = variables_explicatives[tranche_age]
    
    XX = PREDICTEUR2[var_pop]
    YY = PREDICTRICE2[population]
    
    R2_MC = pd.DataFrame([range(0,REPET)])
    
    for ii in range(REPET):
        """
        indexation = sample(list(XX.index),k=len(XX))
        train_index = indexation[0:round(0.8*len(XX))]
        """
        X_train,X_test,Y_train,Y_test = train_test_split(XX,YY,train_size=0.8)
        
        modelk = LinearRegression()
        modelk.fit(X_train,Y_train)
        predk = modelk.predict(X_test)
        
        new_R2 = 1-sum((predk - Y_test)**2)/sum((Y_test.mean()-Y_test)**2)
        R2_MC[ii] = new_R2
    R2_MC = R2_MC.transpose()
    
    R2_hat = R2_hat + [R2_MC.mean()[0]]
R2_hat = pd.DataFrame(R2_hat)
R2_hat.index = list(PREDICTRICE2)
R2_hat.rename(columns={0: 'R2 ajusté'},inplace=True)

print(R2_hat)


"""

###############################################################################

### EXPLORATION DES DONNEES ###################################################

###############################################################################

"""


"""
###############################################################################
### ACP - FONCTIONS #######################################################################
###############################################################################
"""

def circleOfCorrelations(pc_infos, ebouli):
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
        x = pc_infos["PC-0"][idx]
        y = pc_infos["PC-1"][idx]
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(pc_infos.index[idx], xy=(x,y))
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")


def myPCA(df, n_comp, clusters=None):
    # Normalize data
    df_norm = (df - df.mean()) / df.std()
    # PCA
    pca = PCA(n_components=n_comp)
    pca_res = pca.fit_transform(df_norm.values)
    # Ebouli
    ebouli = pd.Series(pca.explained_variance_ratio_)
    ebouli.plot(kind='bar', title="Ebouli des valeurs propres")
    plt.show()
    # Circle of correlations
    # http://stackoverflow.com/a/22996786/1565438
    coef = np.transpose(pca.components_)
    cols = ['PC-'+str(x) for x in range(len(ebouli))]
    pc_infos = pd.DataFrame(coef, columns=cols, index=df_norm.columns)
    circleOfCorrelations(pc_infos, ebouli)
    plt.show()
    # Plot PCA
    dat = pd.DataFrame(pca_res, columns=cols)
    if isinstance(clusters, np.ndarray):
        for clust in set(clusters):
            colors = list("bgrcmyk")
            plt.scatter(dat["PC-0"][clusters==clust],dat["PC-1"][clusters==clust],c=colors[clust])
    else:
        plt.scatter(dat["PC-0"],dat["PC-1"])
    plt.xlabel("PC-0 (%s%%)" % str(ebouli[0])[:4].lstrip("0."))
    plt.ylabel("PC-1 (%s%%)" % str(ebouli[1])[:4].lstrip("0."))
    plt.title("PCA")
    plt.show()
    return pc_infos, ebouli


"""
###############################################################################
### ACP #######################################################################
###############################################################################
"""

variable = "Population_Francaise_Totale"

# ACP - 1
ii = variables_expl[variable]
XX = PREDICTEUR_ACP1[variables_explicatives[ii]]
(pca_pop,eboulis) = myPCA(XX,2)

# ACP - 2
ii = variables_expl[variable]
XX = PREDICTEUR_ACP2[variables_explicatives[ii]]
(pca_pop,eboulis) = myPCA(XX,2)

# ACP - 3
ii = variables_expl[variable]
XX = PREDICTEUR_ACP3[variables_explicatives[ii]]
(pca_pop,eboulis) = myPCA(XX,2)


"""

###############################################################################

### MODELES SUR PROPORTIONS ###################################################

###############################################################################

"""



"""
###############################################################################
### CORRESPONDANCE ############################################################
###############################################################################
"""


correspondance = pd.DataFrame(np.unique(PREDICTEUR["Lieu"]))
correspondance.rename(columns={0: 'Département'},inplace=True)
correspondance.index = range(0,len(correspondance))







"""
###############################################################################
### VARIABLES EXPLICATIVES LOG ################################################
###############################################################################
"""

"""
problème: les p-values rendues sont super élevées (>0.9)
"""
variables_explicatives_log = list()
# evo.R2 <- list()
list_var = list(PREDICTRICE3)
for variable in list_var:
    # variable = "Population_Feminine_0_19"
    # RR <- c()
    # VV <- c()
    XX = PREDICTEUR2
    YY = PREDICTRICE3[variable] 
    n = list(PREDICTEUR2)
    try:
        n = n.remove("Intercept")
    except:
        pass
    
    model1 = Summary_log(XX,YY)
    sum_modelk = model1.sort_values(by="p_values",axis=0)
    
    while((sum_modelk.p_values[-1] > 1e-5 and len(sum_modelk) > 4) or(len(sum_modelk) > 10)):
      pire = Recup_pire_coef_log(sum_modelk)
      # len(pire)
      try:
          n = [x for x in n if x not in pire]
      except:
          pass
      
      # len(n)
      # len(pire)
      XX = PREDICTEUR2[n]
      YY = PREDICTRICE3[variable]
      
      # sum_modelk["p_values"]
      sum_modelk =  Summary_log(XX,YY)[1:].sort_values(by="p_values",axis=0)
      
    n = pd.DataFrame([sum_modelk.index,sum_modelk["p_values"]])
    variables_explicatives_log.append(n)

"""
###############################################################################
### TEST DU MODELE ############################################################
###############################################################################
"""

#Création des jeux de données de tests : 
"""
On va créer ensembles de jeux de données :
    DFtrain contient les 3 jeux de données d'entrainement contenant les années avant celle que l'on veut prédire
    DFtest contient les 3 jeux de données de tests contenant l'année que l'on test : 2013, 2014, 2015
"""

DFtrain = dict()
DFtest = dict()
annees = list([2013,2014,2015])
variablesReponse = list(PREDICTRICE)[2:]
for variableReponse in variablesReponse :
    annees2 = list([2013,2014,2015])
    variablesExpl = variables_explicatives[variables_expl[variableReponse]]
    for annee in annees :
        p0 = PREDICTEUR[-PREDICTEUR["Annee"].isin(annees2)]
        p0 = p0[[c for c in p0.columns if c in variablesExpl]]
        p0 = pd.DataFrame(p0)

        p1 = PREDICTRICE[-PREDICTRICE["Annee"].isin(annees2)]
        p1 = p1[variableReponse]
        p1 = pd.Series(p1)
        nom = str(variableReponse) + "_" + str(annee)
        DFtrain[nom] = pd.concat([p0,p1], axis = 1)
        
        p2 = PREDICTEUR[PREDICTEUR["Annee"] == annee]
        p2 = p2[[c for c in p2.columns if c in variablesExpl]]
        p2 = pd.DataFrame(p2)
        
        p3 = PREDICTRICE[PREDICTRICE["Annee"] == annee]
        p3 = p3[variableReponse]
        p3 = pd.Series(p3)
    
        DFtest[nom] = pd.concat([p2,p3], axis = 1)
        del annees2[0]     
#calcul des prédiction par le modèle choisi précédemment

modelt = dict()
DF_repartition = dict()

variableReponse = variablesReponse[5] 
for annee in annees :
  nom = str(variableReponse) + "_" + str(annee)
  DF_repartition[str(annee)] = DFtest[nom][variableReponse]  
  DF_repartition[str(annee)] = DF_repartition[str(annee)].reset_index()

del variablesReponse[5]

for variableReponse in variablesReponse :
  for annee in annees :
    nom = str(variableReponse) + "_" + str(annee)
    
    X_train = np.array(DFtrain[nom].drop(variableReponse, axis = 1))
    Y_train = np.array(DFtrain[nom][variableReponse])
    X_test = np.array(DFtest[nom].drop(variableReponse, axis = 1))
    Y_test = np.array(DFtest[nom][variableReponse]) 

    modelt[nom] = LinearRegression()
    modelt[nom].fit(X_train,Y_train)
    
    pred = modelt[nom].predict(X_test)
    pred = [x if x>0 else 0 for x in pred ]
    
    
    DFtest[nom] = DFtest[nom].reset_index()
    nem = 'prediction' + str(variableReponse)
    DFtest[nom] = pd.concat([DFtest[nom],pd.Series(pred, name = nem)], axis = 1)
    
    
    df1 = DF_repartition[str(annee)]
    df2 = pd.Series(DFtest[nom][variableReponse])
    df3 = pd.Series(DFtest[nom][nem])
    DF_repartition[str(annee)] = pd.concat([df1,df2, df3], axis = 1)

for annee in annees:
    previsionTotale = 0
    for groupe in variablesReponse:
        previsionTotale = previsionTotale + DF_repartition[str(annee)]['prediction'+groupe]
    previsionTotale = pd.Series(previsionTotale, name = 'previsionTotale')
    DF_repartition[str(annee)] = pd.concat([DF_repartition[str(annee)],previsionTotale], axis = 1)

#représentation graphique :
    
fig = plt.figure(1, figsize=(12, 8))
ptg, = plt.plot(DFtest["Population_Feminine_0_19_2015"].index, DFtest["Population_Feminine_0_19_2015"].Population_Feminine_0_19, 'g.')
plt.xlabel("département")
plt.ylabel("effectifs")
ptr, = plt.plot(DFtest["Population_Feminine_0_19_2015"].index, DFtest["Population_Feminine_0_19_2015"].predictionPopulation_Feminine_0_19, 'r.')
plt.title( "Prévision de la population féminine de 0 à 19 ans par département en 2015")
plt.legend([ptg,ptr],['réel','prévu'])
plt.savefig('popFeminine_test_pred_effectifs.png')
plt.show()





#Calcul du R² total :

def CalculR2(DF_repartition, variablesReponse, annees):
    moyenne = dict()
    R2 = dict()  
    
    
    for annee in annees : 
        moyenne[str(annee)] = DF_repartition[str(annee)].mean(axis = 0) #moyenne de chaque colonne
          
        SCR2 = list()
        SCR = 0
        for groupe in variablesReponse :
            SCR2.append(sum((DF_repartition[str(annee)][groupe]-DF_repartition[str(annee)]['prediction'+groupe])**2))
            SCR = SCR + sum((DF_repartition[str(annee)][groupe]-DF_repartition[str(annee)]['prediction'+groupe])**2)
            
        
          
          
        SCT2 = list()
        SCT = 0
        for groupe in variablesReponse :
            SCT2.append(sum((DF_repartition[str(annee)][groupe]-moyenne[str(annee)][groupe])**2))
            SCT = SCT + sum((DF_repartition[str(annee)][groupe]-moyenne[str(annee)][groupe])**2)
    
        R2[str(annee)] = 1-SCR/SCT
        print('\nSCT = ',SCT," pour l'an ", annee)
        print('SCR = ',SCR," pour l'an ", annee)
        print('R² = ',R2[str(annee)]," pour l'an ", annee)
    return R2
    
R2 = CalculR2(DF_repartition, variablesReponse, annees)

#Calcul des proportions :

for annee in annees: 
  for groupe in variablesReponse:
      #Prévisions
      DF_repartition[str(annee)]['prediction'+groupe] = DF_repartition[str(annee)]['prediction'+groupe]/DF_repartition[str(annee)].previsionTotale
      #Réel
      DF_repartition[str(annee)][groupe] = DF_repartition[str(annee)][groupe]/DF_repartition[str(annee)].Population_Francaise_Totale
      DF_repartition[str(annee)].previsionTotale = 1
      DF_repartition[str(annee)].Population_Francaise_Totale = 1

#représentation graphique
      
fig = plt.figure(1, figsize=(12, 8))
ptg, = plt.plot(DF_repartition["2015"].index, DF_repartition["2015"].Population_Feminine_0_19, 'g.')
plt.xlabel("département")
plt.ylabel("proportion")
ptr, = plt.plot(DF_repartition["2015"].index, DF_repartition["2015"].predictionPopulation_Feminine_0_19, 'r.')
plt.title( "Prévision de la population féminine de 0 à 19 ans par département en 2015")
plt.legend([ptg,ptr],['réel','prévu'])
plt.savefig('popFeminine_test_pred_proportion.png')
plt.show()


#/Calcul du R² pour les proportions :

R2 = CalculR2(DF_repartition, variablesReponse, annees)





"""
###############################################################################
### SVM #######################################################################
###############################################################################
"""

# glmnet ne fonctionne actuellement pas sur windows, nous allons donc prendre
# ici les variables explicatives obtenues sous R


"""
pour PREDICTRICE3:
"""

var_F_0_19 = ["Longueur_Lignes_TGV",
                  "Longueur_Routes_Nationales",
                  "Puissance_Electrique_Hydrolique",
                  "Surface_Aeroports",
                  "Surface_Equipements_Sportifs",
                  "Surface_Prairies",
                  "Surface_Systemes_Ruraux_Complexes",
                  "Surface_Totale_Forets",
                  "Surface_ZIC",
                  "Taux_Couverture_Zones_Inondables",
                  "Temperature_Moyenne_Annuelle"]

var_F_19_39 = ["Etablissements_Industrie_Manufacturiere",
                   "Etablissements_Transports_Entreposage",
                   "Longueur_Reseau_Routier_4_Voies",
                   "Longueur_Routes_Nationales",
                   "Nombre_Exploitations_Bio",
                   "Prix_Vente_Moyen_Apartement",
                   "Quantite_OMR_Collectee",
                   "Surface_Agriculture_Bio",
                   "Surface_Totale",
                   "Surface_Totale_Territoire"]

var_F_40_59 = ["Etablissements_Enseignement",
                   "Etablissements_Industrie_Manufacturiere",
                   "Longueur_Routes_Nationales",
                   "Mediane_Niveau_Vie_Menages",
                   "Nombre_Exploitation_Sans_Signe_quelite",
                   "Nombre_Pecheurs",
                   "Prix_Vente_Moyen_Maison_Individuelle_Neuve",
                   "Surface_Agricole_Totale",
                   "Surface_Espaces_Verts_Urbains",
                   "Surface_Foret_Feuillus",
                   "Surface_Totale_Forets",
                   "Temperature_Moyenne_Annuelle"]

var_F_60_74 = ["Etablissements_Transports_Entreposage",
                   "Longueur_Reseau_Routier_4_Voies",
                   "Nombre_Exploitations_Bio",
                   "Superficie_Totale_Departement",
                   "Surface_Aeroports",
                   "Surface_Espaces_Verts_Urbains",
                   "Surface_Systemes_Ruraux_Complexes",
                   "Surface_Totale",
                   "Surface_Totale_Forets",
                   "Surface_Totale_Territoire",
                   "Surface_ZIC",
                   "Temperature_Moyenne_Annuelle"]

var_F_75 = ["Etablissements_Transports_Entreposage",
                "Longueur_Routes_Nationales",
                "Quantite_OMR_Collectee",
                "Surface_Aeroports",
                "Surface_Agriculture_Bio",
                "Surface_Espaces_Verts_Urbains",
                "Surface_Systemes_Ruraux_Complexes",
                "Surface_Totale",
                "Surface_Totale_Territoire",
                "Surface_ZIC"]

var_M_0_19 = ["Etablissements_Production_Distribution_Electricite_Gaz_Vapeur_Air_Conditionne",
                  "Longueur_Lignes_TGV",
                  "Longueur_Routes_Nationales",
                  "Surface_Aeroports",
                  "Surface_Equipements_Sportifs",
                  "Surface_Systemes_Ruraux_Complexes",
                  "Surface_Totale_Forets",
                  "Surface_ZIC",
                  "Taux_Couverture_Zones_Inondables",
                  "Temperature_Moyenne_Annuelle"]

var_M_19_39 = ["Etablissements_Industrie_Manufacturiere",
                   "Etablissements_Transports_Entreposage",
                   "Longueur_Routes_Nationales",
                   "Nombre_Exploitations_Bio",
                   "Prix_Vente_Moyen_Apartement",
                   "Superficie_Espaces_Proteges",
                   "Surface_Agriculture_Bio",
                   "Surface_Totale",
                   "Surface_Totale_Territoire",
                   "Temperature_Moyenne_Annuelle"]

var_M_40_59 = ["Autre_Etablissements_Services",
                   "Etablissements_Industrie_Manufacturiere",
                   "Longueur_Reseau_Routier_4_Voies",
                   "Longueur_Routes_Nationales",
                   "Mediane_Niveau_Vie_Menages",
                   "Nombre_Annuel_Jours_Gel",
                   "Nombre_Exploitation_Sans_Signe_quelite",
                   "Nombre_Total_Exploitation",
                   "Prix_Vente_Moyen_Maison_Individuelle_Neuve",
                   "Quantite_OMR_Collectee",
                   "Surface_Agricole_Totale",
                   "Surface_Espaces_Verts_Urbains",
                   "Taux_Chomage"]

var_M_60_74 = ["Etablissements_Transports_Entreposage",
                   "Longueur_Reseau_Routier_4_Voies",
                   "Longueur_Routes_Nationales",
                   "Nombre_Exploitations_Bio",
                   "Puissance_Electrique_Photovoltaique",
                   "Surface_Aeroports",
                   "Surface_Agriculture_Bio",
                   "Surface_Systemes_Ruraux_Complexes",
                   "Surface_Totale_Forets",
                   "Surface_ZIC"]

var_M_75 = ["Etablissements_Transports_Entreposage",
                "Longueur_Routes_Nationales",
                "Quantite_OMR_Collectee",
                "Surface_Aeroports",
                "Surface_Agriculture_Bio",
                "Surface_Totale",
                "Surface_Totale_Territoire",
                "Surface_ZIC",
                "Taux_Couverture_Zones_Inondables",
                "Temperature_Moyenne_Annuelle"]

var_expl = [var_F_0_19,var_F_19_39,var_F_40_59,var_F_60_74,var_F_75,
            var_M_0_19,var_M_19_39,var_M_40_59,var_M_60_74,var_M_75]

list_var = list(PREDICTRICE3)
R2_SVM = list()
for variable in list_var:
    # variable = "Population_Masculine_19_39"
    ii = variables_expl_3[variable]
    nn = var_expl[ii]
    
    XX = PREDICTEUR2[nn]
    YY = PREDICTRICE3[variable]
    
    try:
        n = n.remove("Intercept")
    except:
        pass
    
    X_train,X_test,Y_train,Y_test = train_test_split(XX,YY,train_size=0.8)
    
    # model1
    model1 = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    model1.fit(X_train,Y_train)
    pred1 = model1.predict(X_test)
    new_R2 = 1-sum((pred1 - Y_test)**2)/sum((Y_test.mean()-Y_test)**2)
    
    """
    fig = plt.figure(1, figsize=(15,8))
    green_dot, = plt.plot(range(len(Y_test)),Y_test,'g.', markersize=3)
    red_dot, = plt.plot(range(len(Y_test)),pred1,'r.', markersize=3)
    
    plt.title("Prédiction sur Population_Masculine_19_39 avec les SVM")
    
    plt.legend([red_dot, green_dot], ["réel", "prévision"])
    
    
    figname = "P:/4sci01/Projet_Lardjane_R/images_Baptiste/plot_SVM.png"
    plt.savefig(figname)
    
    plt.show()
    """
    
    R2_SVM = R2_SVM + [new_R2]
    
R2_SVM = pd.DataFrame(R2_SVM)
R2_SVM.index = list(PREDICTRICE3)
R2_SVM.rename(columns={0: 'R2 ajusté'},inplace=True)

for variable in list_var:
    ii = variables_expl_3[variable]
    print(variable)
    print(var_expl[ii])


"""
###############################################################################
### RANDOM FORREST ############################################################
###############################################################################
"""

list_var = list(PREDICTRICE3)
R2_RF = list()
for variable in list_var:
    # variable = "Population_Masculine_19_39"
    ii = variables_expl_3[variable]
    nn = var_expl[ii]
    
    XX = PREDICTEUR2[nn]
    YY = PREDICTRICE3[variable]
    
    try:
        n = n.remove("Intercept")
    except:
        pass
    
    X_train,X_test,Y_train,Y_test = train_test_split(XX,YY,train_size=0.8)
    
    # model1
    model1 = RandomForestRegressor()
    model1.fit(X_train,Y_train)
    pred1 = model1.predict(X_test)
    new_R2 = 1-sum((pred1 - Y_test)**2)/sum((Y_test.mean()-Y_test)**2)
    
    """
    fig = plt.figure(1, figsize=(15,8))
    green_dot, = plt.plot(range(len(Y_test)),Y_test,'g.', markersize=3)
    red_dot, = plt.plot(range(len(Y_test)),pred1,'r.', markersize=3)
    
    plt.title("Prédiction sur Population_Masculine_19_39 avec les Random Forest")
    
    plt.legend([red_dot, green_dot], ["réel", "prévision"])
    
    
    figname = "P:/4sci01/Projet_Lardjane_R/images_Baptiste/plot_RF.png"
    plt.savefig(figname)
    
    plt.show()
    """
    
    
    R2_RF = R2_RF + [new_R2]
    
R2_RF = pd.DataFrame(R2_RF)
R2_RF.index = list(PREDICTRICE3)
R2_RF.rename(columns={0: 'R2 ajusté'},inplace=True)

"""
###############################################################################
### GRADIENT BOSTING ##########################################################
###############################################################################
"""

list_var = list(PREDICTRICE3)
R2_GB = list()
for variable in list_var:
    # variable = "Population_Masculine_19_39"
    ii = variables_expl_3[variable]
    nn = var_expl[ii]
    
    XX = PREDICTEUR2[nn]
    YY = PREDICTRICE3[variable]
    
    try:
        n = n.remove("Intercept")
    except:
        pass
    
    X_train,X_test,Y_train,Y_test = train_test_split(XX,YY,train_size=0.8)
    
    # model1
    model1 = GradientBoostingRegressor()
    model1.fit(X_train,Y_train)
    pred1 = model1.predict(X_test)
    new_R2 = 1-sum((pred1 - Y_test)**2)/sum((Y_test.mean()-Y_test)**2)
    
    """
    """
    fig = plt.figure(1, figsize=(15,8))
    
    green_dot, = plt.plot(range(len(Y_test)),Y_test,'g.', markersize=3)
    red_dot, = plt.plot(range(len(Y_test)),pred1,'r.', markersize=3)
    
    plt.title("Prédiction sur Population_Masculine_19_39 avec Gradient Boosting")
    
    plt.legend([red_dot, green_dot], ["réel", "prévision"])
    
    
    figname = "P:/4sci01/Projet_Lardjane_R/images_Baptiste/plot_GB.png"
    plt.savefig(figname)
    
    plt.show()
    """
    """
    
    R2_GB = R2_GB + [new_R2]
    
R2_GB = pd.DataFrame(R2_GB)
R2_GB.index = list(PREDICTRICE3)
R2_GB.rename(columns={0: 'R2 ajusté'},inplace=True)


