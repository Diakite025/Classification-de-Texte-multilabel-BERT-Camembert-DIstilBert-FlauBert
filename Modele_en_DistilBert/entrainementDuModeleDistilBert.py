
from transformers import AutoTokenizer, DistilBertModel
import torch.optim as optim

import torch
import torch.nn as nn

import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pretraitementDesDonnees
import importEtInsert_data
import utility
import time


        



class DistilBertClass(torch.nn.Module):
    
    """
        Initialise la class  DistilBert pour la classification de texte.

        Args:
            num_classes (int): Le nombre de classes pour la classification (par défaut: 51).
    """
    
    def __init__(self,num_classes=51):
        super(DistilBertClass, self).__init__()
        self.DistilBert =DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_classes)
       
        

    def forward(self, input_ids, attn_mask):
        output = self.DistilBert(input_ids, attention_mask=attn_mask)
        
        hidden_state = output.last_hidden_state[:, 0]
        pooler=self.pre_classifier(hidden_state)
        pooler_probabilities=torch.sigmoid(pooler)
        output_dropout = self.dropout(pooler_probabilities)
        output = self.linear(output_dropout)
        return output
    
    



class ModelDistilBert:
    
    def __init__(self,df_train,df_valid,target_list,path):
        
        """
        Description : Cette classe permet de créer et d'entraîner un modèle DistilBert.

        Args:
            df_train (DataFrame): Données d'entraînement.
            df_valid (DataFrame): Données de validation.
            target_list (list)  : Liste des noms des classes (liste des catégories).
            path (str)          : Chemin vers le fichier où le modèle sera sauvegardé.
        """
        
        self.MAX_LEN = 256
        self.batch_size=32
        self.EPOCHS = 5
        self.LEARNING_RATE = 1e-05
        self.patience = 3
        self.THRESHOLD = 0.5 # seuil de classification
        
        self.path=path
        self.target_list=target_list
        self.nb_classes=len(target_list)
        self.df_train=df_train
        self.df_valid=df_valid
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
       


    def preparation_des_données(self):
            
        train_dataset =pretraitementDesDonnees.PreparationDesDonnees(self.df_train, self.tokenizer, self.MAX_LEN, self.target_list)
        valid_dataset = pretraitementDesDonnees.PreparationDesDonnees(self.df_valid, self.tokenizer, self.MAX_LEN, self.target_list)
                        

        
                    
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )

        self.val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

       
    
    def Création_model(self):
            
        self.model=DistilBertClass(self.nb_classes)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE )

            
    def loss_fn(self,outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)


 
            
            
                


    def train_model(self):
        
        """
        Description :   La fonction "train_model" permet d'entraîner le modèle pour une époque.
                        Elle effectue les différentes étapes d'entraînement :
                    
                        - forward : la passe avant pour obtenir les prédictions et calculer la perte.
                        - backward : la rétropropagation pour mettre à jour les poids du modèle à partir de la perte(loss) calculée.
                        
        Retourne    :   La fonction "train_model" renvoie la Accuracy et la perte(loss) des données d'entraînement pour une époque .
        """

        
        losses = []
        correct_predictions = 0
        num_samples = 0
        
        # Met le modèle en mode entraînement
        self.model.train()
        
        # Boucle avec une barre de progression pour suivre l'évolution du modèle (loss, accuracy, le temps estimé pour une époque)
        loop = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader), leave=True, colour='BLUE')

        for batch_idx, data in loop:
            ids = data['input_ids'].to(self.device, dtype=torch.long)
            mask = data['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            # forward
            outputs = self.model(ids, mask) 
            loss = self.loss_fn(outputs, targets) # Calcule la perte (loss)
            losses.append(loss.item())
            
            # Application de la fonction sigmoïde à la sortie du modèle pour calculer l'appartenance  d'un texte à chaque catégorie.
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs == targets)
            num_samples += targets.size   

            # backward
            self.optimizer.zero_grad()
            loss.backward() # Calcule des gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clipping des gradients pour normaliser et éviter l'explosion des gradients

            self.optimizer.step()

            # Met à jour la description de la barre de progression avec la perte(loss) et la Accuracy actuelle
            loop.set_description(f"Loss: {loss.item():.4f}, Accuracy: {float(correct_predictions) / num_samples:.4f}")

        # Calcule et retourne la Accuracy et la perte moyenne pour cette époque
        accuracy = float(correct_predictions) / num_samples if num_samples > 0 else 0.0
        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        return accuracy, mean_loss





    def eval_model(self):
        
        """
        Description :   La fonction "eval_model" permet d'évaluer le modèle sur les données de validation pour une époque.
 
                        
        Retourne    :   La fonction "eval_model" renvoie la Accuracy et la perte(loss) des données de validation pour une époque.
        """
        
        losses = []
        correct_predictions = 0
        num_samples = 0
            
        self.model.eval()

        with torch.no_grad():
            # Crée une boucle avec une barre de progression pour les batches de validation
            loop = tqdm(enumerate(self.val_data_loader), total=len(self.val_data_loader), leave=True, colour='BLUE')
            
            for batch_idx, data in loop:
                ids = data['input_ids'].to(self.device, dtype = torch.long)
                mask = data['attention_mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
            
            
                outputs =  outputs = self.model(ids, mask) 

                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())

                
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
                targets = targets.cpu().detach().numpy()
                correct_predictions += np.sum(outputs==targets)
                num_samples += targets.size  
                 
        # retourne la l'Accuracy et la perte(loss)
        return float(correct_predictions)/num_samples, np.mean(losses)
            
            
        
    def EntrainementModele(self):
                
        history = defaultdict(list)  # Pour stocker l'historique des métriques
        epochs_no_improve = 0        # Compteur pour le nombre d'époques sans amélioration
        best_loss = float('inf')     # Meilleure perte de validation observée
        
        
        time.sleep(3)
      

    

        # Boucle d'entraînement pendant self.EPOCHS 
        for epoch in range(1, self.EPOCHS+1):
            start_time = time.time()
            print(f'Epoch {epoch}/{self.EPOCHS}')
            
            train_acc, train_loss = self.train_model()  # Entraîne le modèle et obtient les métriques d'entraînement
            val_acc, val_loss = self.eval_model()       # Évalue le modèle et obtient les métriques de validation

            # Vérifie si la loss de validation s'est améliorée
            if val_loss < best_loss:
                
                best_loss = val_loss
                epochs_no_improve = 0
                
                # Sauvegarde le modèle si la loss de la validation actuel est meilleure que la précédente
                torch.save(self.model.state_dict(), self.path)
                
            else:
                epochs_no_improve += 1

            # Arrêt précoce si le modèle ne s'est pas amélioré depuis un certain nombre d'époques (patience par défaut 3). On parle de "l'early stopping".
            if epochs_no_improve >= self.patience:
                print(f'\nEarly stopping après {self.patience} epoch sans amélioration du modèle.')
                end_time = time.time()
                execution_time = end_time - start_time
    
                print("\nTemps d'exécution:", execution_time/60, "min")
                print(f'\ntrain_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

                break
            
            
            end_time = time.time()
            execution_time = end_time - start_time
    
            print("\nTemps d'exécution:", execution_time/60, "min")
            print(f'\ntrain_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

            # Stocke les métriques d'entraînement et de validation 
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

        
            # Enregistre l'historique d'entraînement dans un fichier JSON
        with open('train_history.json', 'w') as json_file:
            json.dump(history, json_file)

    
        
                

def main(lien_collection_train="df_train_en",lien_collection_valid="df_valid_en",target_list= utility.Liste_categories(),path="BestModel/DistilBertModel.pth"):
    
    """
        Description : Cette fonction execute l'entrainement du modèle

        Args:
            lien_collection_train (str) : Lien de de la collection des données d'entraînement.
            lien_collection_valid (str) : Lien de de la collection des données de validation.
            path (str)          : Chemin vers le fichier où le modèle sera sauvegardé.
        """
    
    
    
    print("\n ################################# Importation des données ##################################### \n")
    
    df_train=importEtInsert_data.Importation_dataFrame(lien_collection_train)
    df_valid = importEtInsert_data.Importation_dataFrame(lien_collection_valid)
    
    print(f"Train: {df_train.shape}, Valid: {df_valid.shape}")
   
    
    print("\n ################################### Entrainement du modèle ############################## \n")
    
    modelDistilBert=ModelDistilBert(df_train,df_valid,target_list,path)
    
    modelDistilBert.preparation_des_données()
    modelDistilBert.Création_model()
    modelDistilBert.EntrainementModele()
         
                
                



if __name__ == '__main__':
    
    main()