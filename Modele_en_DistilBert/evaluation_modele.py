
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report

from transformers import AutoTokenizer
import torch
import pretraitementDesDonnees
import entrainementDuModeleDistilBert ,importEtInsert_data,utility
import pandas as pd
from tqdm import tqdm



class EvaluationModel():
    
    """
 
        Description : Classe d'évaluation du modèle de classification fine-tuning basé sur BERT.
    
        Args:
            df_test (DataFrame): Dataframe de test.
            target_list (list) : Liste des noms des classes (liste des catégories).
            path (str)         : Chemin vers le modèle entraîné.
        """
    
    def __init__(self,df_test,target_list,path="BestModel/model_en_V2.pth"):
        
        self.batch_size=32
        self.MAX_LEN = 256
        self.target_list=target_list
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Chargement du modèle entraîné depuis le chemin spécifié dans l'argument path
        model=entrainementDuModeleDistilBert.DistilBertClass()
        model.eval()
        model.load_state_dict(torch.load(path))
        self.model = model.to(self.device)
        
        # Prétraitement des données de test
        test_dataset = pretraitementDesDonnees.PreparationDesDonnees(df_test, self.tokenizer, self.MAX_LEN, self.target_list)
        self.test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,shuffle=False,num_workers=0)
    
    
        # Evaluation du modèle
        self.get_predictions()
        
        
    def get_predictions(self):
       
        """
        Cette fonction calcule les prédictions du modèle sur les données de test.
        """
        self.model = self.model.eval()
        
        textes = []
        predictions = []
        prediction_probs = []
        target_values = []

        loop = tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader), leave=True, colour='BLUE')

        with torch.no_grad():
            for batch_idx, data in loop:
                texte = data["texte"]
                ids = data["input_ids"].to(self.device, dtype = torch.long)
                mask = data["attention_mask"].to(self.device, dtype = torch.long)
                targets = data["targets"].to(self.device, dtype = torch.float)
                
                outputs = self.model(ids, mask)
                outputs = torch.sigmoid(outputs).detach().cpu()
                preds = outputs.round()
                targets = targets.detach().cpu()

                textes.extend(texte)
                predictions.extend(preds)
                prediction_probs.extend(outputs)
                target_values.extend(targets)
        
        self.predictions = torch.stack(predictions)
        prediction_probs = torch.stack(prediction_probs)
        self.target_values = torch.stack(target_values)
        
        

        # Génération des courbes ROC et rapport de classification
        self.courbe_roc()
        self.rapport_classification()


    def courbe_roc(self):
        
        """
        Cette fonction génère et affiche les courbes ROC pour évaluer la performance du modèle.
        """
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(self.target_list)):
            fpr[i], tpr[i], _ = roc_curve(self.target_values[:, i], self.predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    
        fpr["micro"], tpr["micro"], _ = roc_curve(self.target_values.ravel(), self.predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        
        plt.figure(figsize=(20, 18))


        for i in range(len(self.target_list)):
            plt.plot(fpr[i], tpr[i], label='ROC curve (class {}) (area = {:.2f})'.format(self.target_list[i], roc_auc[i]))


        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()


    def rapport_classification(self):
    
        """
        Cette fonction affiche le rapport de classification pour évaluer la performance du modèle.
        """
        print(classification_report(self.target_values, self.predictions, target_names=self.target_list))
        
        
def main(df: pd.DataFrame = None,target_list:list = utility.Liste_categories(),lien_collection_test="df_test_en",path="BestModel/DistilBertModel.pth"):
    
    
    """
    Fonction principale pour évaluer le modèle de catégorisation d'annonces.
    
    Arguments :
        df  (pd.DataFrame, optionnel)        : DataFrame contenant les données à utiliser pour l'évaluation. Si None, les données seront importées de MongoDB.
        lien_collection_test (str, optionnel): Nom de la collection MongoDB à partir de laquelle importer les données de test si df est None.
                                               Par défaut "test_multlang".
        path (str, optionnel)                : Chemin du fichier contenant le modèle entraîné à utiliser pour l'évaluation.
                                               Par défaut "BestModel/DistilBertModel.pth".
        target_list (list)                   : Liste des noms des classes (liste des catégories).

    """
    
    print("\n ################################# Importation des données ##################################### \n")
    
    if df is not None:
        
        EvaluationModel(df, target_list, path)
    else:
        df_test = importEtInsert_data.Importation_dataFrame(lien_collection_test)
        EvaluationModel(df_test, target_list, path)
        
    
    
if __name__ == '__main__':
    main()
    