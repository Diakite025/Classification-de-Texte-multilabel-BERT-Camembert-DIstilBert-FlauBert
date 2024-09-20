import json
from elasticsearch import Elasticsearch
import elasticsearch.helpers
from pymongo import MongoClient
import pandas as pd
import utility 



def DeElasticSearch_A_MongoDB(collection:str, requete:str, lien_BD_Elasticsearch:str, lien_BD_MongoDB):
    
    """
    Description : Cette fonction importe des données depuis Elasticsearch et les stocke dans MongoDB.
    
    Arguments :
    - collection : La collection où les données seront insérées dans MongoDB.
    - requete : Le chemin du fichier contenant la requête Elasticsearch pour récupérer les données.
    - lien_BD_Elasticsearch : Le lien vers la base de données Elasticsearch à partir de laquelle les données seront récupérées.
    - lien_BD_MongoDB : Le lien vers la base de données MongoDB où les données seront insérées.
    
    post : Données stockées dans la collection spécifiée.
    """

    # Connexion à la base de données Elasticsearch
    es = Elasticsearch([lien_BD_Elasticsearch], timeout=600)
    
    # Ouvrir et charger la requête Elasticsearch à partir du fichier
    fdr = open(requete, "r")
    request = json.load(fdr)
    
    # Connexion à la base de données MongoDB
    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning

    # Récupéreration des données depuis Elasticsearch et les insérer dans la base de données MongoDB
    res = elasticsearch.helpers.scan(
        es,
        index="idx*",
        preserve_order=True,
        query=request,
        raise_on_error=None,
        request_timeout=1200
    )

    for hit in res:
        dic_data = {}
        
        # Récupération des champs qui nous intéressent 
        dic_data["Text"] = hit["_source"]["title"] + " "+ hit["_source"]["description"]
        dic_data["textes"] = hit["_source"]["full_text"]
        dic_data["categories"] = hit["_source"]["options"]["kw_arr_categories"]
        
        
        # Insertion des données dans la collection MongoDB
        ml_db[collection].insert_one(dic_data)




def Creation_data_frame(collection_source:str, collection_dest:str,lien_BD_MongoDB,liste_categories=utility.Liste_categories()):
    
    
    """
    Description : Fonction pour créer un DataFrame multi-étiquettes à partir des données de la collection source MongoDB 
                  et l'insérer dans la collection de destination.
    
    Paramètres :
    - collection_source (str) : Nom de la collection source dans MongoDB.
    - collection_dest (str)   : Nom de la collection de destination dans MongoDB.
    - liste_categories (list) : Liste des catégories.
    - lien_BD_MongoDB (str)   : Lien de connexion à la base de données MongoDB.
    """

    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning
    
    for doc in ml_db[collection_source].find():
        
        text = doc["textes"]
        titre = doc["Titre"]
        description = doc["Description"]
        categories = set(doc["categories"])
        
        # Création un vecteur d'étiquettes pour chaque catégorie
        label_vector = [1 if label in categories else 0 for label in liste_categories]
        
     
        data_dict = {
            "Texte":text,
            "Titre": titre,
            "Description": description,
        }
        
        # Ajouter les étiquettes au dictionnaire de données
        for i, label in enumerate(liste_categories):
            data_dict[label] = label_vector[i]
        
        # Insertion des documenst
        ml_db[collection_dest].insert_one(data_dict)



def Importation_dataFrame(nom_collection:str,lien_BD_MongoDB):
    
    
    """
    Description: Fonction pour importer les données de la collection MongoDB spécifiée dans un DataFrame.

    Paramètres :
    - nom_collection (str)  : Le nom de la collection à partir de laquelle importer les données.
    - lien_BD_MongoDB (str) : Le lien vers la base de données MongoDB où se trouve la collection.

    Retourne :
    - data_frame (pd.DataFrame) : Le DataFrame contenant les données extraites de la collection.
    """
    
    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning
    

    collection = ml_db[nom_collection]
    documents_list = []
    
 
    for document in collection.find():
        documents_list.append(document)
        

    data_frame = pd.DataFrame(documents_list)
    
    if '_id' in data_frame.columns:
        data_frame=data_frame.drop(columns="_id")
    
    data_frame=data_frame[(data_frame.iloc[:, -51:-1] != 0).any(axis=1)]
    
    return data_frame


def Supprime_collection(liste_nom_collection:list,lien_BD_MongoDB):
    
    """
    Description: Fonction pour supprimer les collections à aprtir d'une liste de nom de collection dans la base de données MongoDB.

    Paramètres :
    - liste_nom_collection (list) : Liste des noms de collections à supprimer.
    - lien_BD_MongoDB (str)       : Le lien vers la base de données MongoDB où se trouvent les collections.
 
    """
    
    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning
    
    for collection in liste_nom_collection:
        if collection in ml_db.list_collection_names():
            ml_db[collection].drop()
            print(f"La collection '{collection}' a été supprimée avec succès.")
        else:
            print(f"La collection '{collection}' n'existe pas dans la base de données.")
            
            
            
            
            

def insertion_MDB(collection,df,lien_BD_MongoDB):

    """
    Description: Fonction pour insère les données d'un DataFrame dans une collection MongoDB.

    Paramètres :
    - collection (str)      : Le nom de la collection où insérer les données.
    - df (pd.DataFrame)     : Le DataFrame contenant les données à insérer.
    - lien_BD_MongoDB (str) : Le lien vers la base de données MongoDB où se trouve la collection.


    """

    ml_mgdb = MongoClient(lien_BD_MongoDB)
    ml_db = ml_mgdb.machine_learning
    
    
    for index, row in df.iterrows():
        data_dict = row.to_dict()
        ml_db[collection].insert_one(data_dict)




