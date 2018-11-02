# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

@author: BAPTISTE BOUILLON
"""

import os as os
os.chdir("P:/4sci01/Projet_Lardjane_R/Python/")
from FONCTIONS_BAPTISTE import *



path = "P:/4sci01/Projet_Lardjane_R/DATA_verifiees/predicteur_tmp/"
os.chdir(path)

path2 = "P:/4sci01/Projet_Lardjane_R/DATA_verifiees/predicteur_nettoyé/"


for file in glob.glob("*.csv"):
    DATA = pd.read_csv(file,sep=";",encoding="Latin-1")
    
    DATA_FINAL = pd.DataFrame([],columns=np.unique(DATA["Série"]))
    List_dep = np.unique(DATA["Lieu"])
    
    for dep in List_dep:
        DATA_TMP = DATA[DATA["Lieu"]==dep].iloc[:,2:].drop(DATA.columns[3],axis=1)
        test = transpose_DATA(dep,DATA_TMP)
        
        DATA_FINAL = DATA_FINAL.append(test)
    
    file_name = path2 + file[:-4]+"_sorted.csv"
    DATA_FINAL.to_csv(file_name,sep=";",index=False,encoding="Latin-1")





