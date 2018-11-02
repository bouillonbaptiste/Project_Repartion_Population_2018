# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:43:03 2018

@author: BAPTISTE BOUILLON
"""

import os as os
os.chdir("P:/4sci01/Projet_Lardjane_R/Python/")
from FONCTIONS_BAPTISTE import *

path = "P:/4sci01/Projet_Lardjane_R/DATA_verifiees/a_predire_nettoyé/"
os.chdir(path)



"""
Partie imputation
"""
### pour chaque fichier
file = 0
ttfiles = glob.glob("*.csv")


while file<len(ttfiles):
    
    print("---",round(file/29*100,2)," %  :", ttfiles[file])
    
    # DATA = pd.read_csv(ttfiles[file],sep=";",encoding="Latin-1",na_values="ND",decimal=",")
    DATA = pd.read_csv("PopulationFrancaise_sorted.csv",sep=";",encoding="Latin-1",na_values="ND",decimal=",")
    
    Cols = DATA.columns
    newCols = Cols[~Cols.isin(["index","Lieu"])]
    
    annee_debut = 1995
    annee_fin = 2015
    ### Pour chaque variable dans le fichier
    variablei = 0
    for variable in newCols:
        
        print("-------------",round(variablei/len(newCols)*100,2)," % : ",variable)
        
        DATA_k = pd.DataFrame({"Lieu":DATA["Lieu"],
                               "Année":DATA["index"],
                               variable:DATA[variable]})
        
        List_Pays = np.unique(DATA_k["Lieu"])
        DATA_FINAL = pd.DataFrame([],columns=DATA_k.columns)
        for dep in List_Pays:
            retour = rajoutAnnee(DATA_k[DATA_k["Lieu"]==dep],annee_debut,annee_fin,dep,variable)
            
            # tableau = DATA_k[DATA_k["Lieu"]==dep]
            # len(retour2)
            retour2 = imputOpti(retour[variable],annee_debut,annee_fin)
            retour3 = recreerDep(retour2,dep,annee_debut,annee_fin,variable)
            
            DATA_FINAL = DATA_FINAL.append(retour3)
        
        filename = "C:/Users/Baptiste/Desktop/Projet_Lardjane_R/DATA_verifiees/a_predire_imputé/" + variable + ".csv"
        DATA_FINAL.to_csv(filename,index=False,sep=";",encoding="Latin-1")
        
        variablei += 1
        
        
    print("------------- 100.0 %")
    file += 1

print("terminé !")


"""
Partie regroupement de toutes les variables
"""


path = "C:/Users/Baptiste/Desktop/Projet_Lardjane_R/DATA_verifiees/a_predire_imputé/"
os.chdir(path)


ttfiles = glob.glob("*.csv")
DATA_TMP = pd.read_csv(ttfiles[0],sep=";",encoding="Latin-1")

DATA_FINAL_CONCAT = pd.DataFrame({"Année":DATA_TMP["Année"],
                                  "Lieu":DATA_TMP["Lieu"]})

for file in ttfiles:
    DATA_TMP = pd.read_csv(file,sep=";",encoding="Latin-1")
    DATA_FINAL_CONCAT = DATA_FINAL_CONCAT.join(DATA_TMP[file[:-4]],how="outer")


filename = "C:/Users/Baptiste/Desktop/Projet_Lardjane_R/DATA_verifiees/PREDIRE.csv"
DATA_FINAL_CONCAT.to_csv(filename,index=False,sep=";",encoding="Latin-1")


"""
# ne garder que les colonnes qui nous intéressent.

import os as os
os.chdir("P:/4sci01/Projet_Lardjane_R/Python/")
from FONCTIONS_BAPTISTE import *

path = "P:/4sci01/Projet_Lardjane_R/DATA_verifiees/"
os.chdir(path)

DATA = pd.read_csv("DONNEES_ALL.csv",sep=";",encoding="Latin-1")

names_DATA_INTERDIT =["Nombre_Emploi_Agriculture_Peche",
                      "Nombre_Emploi_Construction",
                      "Nombre_Emploi_Tertiaire",
                      "Nombre_Emploi_Total",
                      "Nombre_Emploi_Industrie",
                      "Nombre_menages",
                      "Nombre_menages_0_voiture",
                      "Nombre_menages_1_voiture",
                      "Nombre_menages_plus_1_place_stationnement",
                      "Nombre_menages_plus_2_voiture",
                      "Nombre_menages_plus_3_voiture",
                      "Nombre_Residences_Principales_Cantons_Littoraux",
                      "Nombre_Residences_Principales_Communes_Littorales",
                      "Nombre_Residences_Secondaires_Cantons_Littoraux",
                      "Nombre_Residences_Secondaires_Communes_Littorales",
                      "Nombre_RP",
                      "Nombre_RP_Chauffage_Autre",
                      "Nombre_RP_Chauffage_Autres_Combustibles",
                      "Nombre_RP_Chauffage_Central_Collectif",
                      "Nombre_RP_Chauffage_Central_Individuel",
                      "Nombre_RP_Chauffage_Electrique",
                      "Nombre_RP_Chauffage_Fioul",
                      "Nombre_RP_Chauffage_Gaz_Bouteille_Citerne",
                      "Nombre_RP_Chauffage_Gaz_Ville_Reseau",
                      "Nombre_RP_Chauffage_Tout_Electrique",
                      "Nombre_RP_Chauffage_Urbain",
                      "Residences_Principales",
                      "Residences_Secondaires",
                      "Total_Logement",
                      "Logement_Maison",
                      "Logement_Apartement",
                      "Autre_Logement",
                      "Surface_Totale_Terres_Artificielles",
                      "Nombre_Logements_Appartement_Cantons_Littoraux",
                      "Nombre_Logements_Appartement_Communes_Littorales",
                      "Nombre_Logements_Cantons_Littoraux",
                      "Nombre_Logements_Communes_Littorales",
                      "Nombre_Logements_Maison_Cantons_Littoraux",
                      "Nombre_Logements_Maison_Communes_Littorales"]


DATA_new = DATA[DATA.columns[~DATA.columns.isin(names_DATA_INTERDIT)]]

filename = "P:/4sci01/Projet_Lardjane_R/DATA_ALL2.csv"
DATA_new.to_csv(filename,index=False,sep=";",encoding="Latin-1")

"""






















