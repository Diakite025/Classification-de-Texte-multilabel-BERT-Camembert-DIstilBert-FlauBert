import pandas as pd
from deep_translator import GoogleTranslator
from pymongo import MongoClient
import utility



def Undersampling(df, valeur, categorie_a_aumenter):
    
    """
    Description: Fonction qui sous-échantillonne un DataFrame afin de réduire le nombre d'exemples  des catégories qui ont un excès d'échantillons.

    Paramètres :
    - df (pd.DataFrame)          : Le DataFrame contenant les données.
    - valeur (int)               : Le nombre d'exemples à conserver.
    - categorie_a_aumenter (str) : Le nom des catégories à ne pas sous-échantillonner.

    Retourne   :
    - df (pd.DataFrame) : Le DataFrame sous-échantillonné.
    """

    
    count = valeur
    for i in df.columns[2:]:
        mask = (df[i] == 1) & (df[categorie_a_aumenter] == 0).all(axis=1)
        exemples_selectionnes = df.loc[mask]
        size = len(exemples_selectionnes)
        if size > count:
            lignes_a_supprimer = exemples_selectionnes.sample(n=size - count)
            index = list(lignes_a_supprimer.index)
            df = df.drop(index)
    return df

    




def traduire_texte(texte, source_lang='fr', target_lang='en'):
    
    """
    Description : Cette fonction crée une copie d'un texte à partir de sa traduction.

    Paramètres  :
    - texte (str)       : Le texte à traduire.
    - source_lang (str) : La langue source du texte (par défaut anglais).
    - target_lang (str) : La langue cible de la traduction (par défaut français).

    Retourne :
    - traduction (str)  : La copie du texte original.
    """
    
    translator_en_fr = GoogleTranslator(source=target_lang, target=source_lang)
    translator_fr_en = GoogleTranslator(source=source_lang, target=target_lang)
    
    if len(texte) < 4000:
       
   
        traduction_fr_en = translator_fr_en.translate(texte)
        traduction_en_fr = translator_en_fr.translate(traduction_fr_en)
    
            
    else:
        return "Non"
            

    
    return traduction_en_fr






def Oversampling(data: pd.DataFrame, count, collection: str,lien_BD_MongoDB,name_categories=utility.Liste_categories(), source_lang='fr', target_lang='en'):
    
    """
    Description : Cette fonction permet de créer des données syntetyques 
                    pour les classe minoritaire (sur-échantillonnage ) et stocke  les nouvelle données créé dans une collection.

    Paramètres  :
    - data (pd.DataFrame)    : Le DataFrame contenant les données d'origine.
    - name_categories (list) : Liste des noms de catégories .
    - count (int)            : Le nombre maximale d'exemples à sur-échantillonner pour chaque catégorie.
    - collection (str)       : Le nom de la collection où insérer les données sur-échantillonnées.
    - lien_BD_MongoDB (str)  : Le lien vers la base de données MongoDB.
    - source_lang (str)      : La langue source des textes (par défaut anglais).
    - target_lang (str)      : La langue cible des traductions (par défaut français).

    """

    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning
    
  
    df = data.copy()

    # Parcours de chaque catégorie à sur-échantillonner
    for label in name_categories:
        # Sélection des exemples appartenant à la catégorie actuelle
        exemples_cat = df[df[label] == 1]
        
        # Vérification si le nombre d'exemples de la catégorie est inférieur au nombre maximal d'echantillon(count)
        if count > len(exemples_cat):
            # Calcul du nombre d'exemples à sur-échantillonner pour cette categorie
            N = count - len(exemples_cat)
            
            # Sélection aléatoire d'exemples de la categorie
            exemples_selectionnes = exemples_cat.sample(n=N, replace=True)
            
            # Parcours de chaque exemple sélectionné pour créer une copie par traduction
            for _, row in exemples_selectionnes.iterrows():
                traductions = traduire_texte(row["Text"], source_lang, target_lang)
                N = N - 1
                
                print("##################" + label + str(N) + " ###########################")
                
                
                if traductions != "Non":
                    row["Text"] = traductions
                    
                    ml_db[collection].insert_one(row.to_dict())
    


