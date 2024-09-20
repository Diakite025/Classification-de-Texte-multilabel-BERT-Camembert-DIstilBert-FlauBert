
import torch
from torch import nn
from transformers import  AutoTokenizer, FlaubertModel





class FlauBertClass(nn.Module):
    def __init__(self, num_classes=51):
        """
        Initialise la classe BERT pour la classification de texte.

        Args:
            num_classes (int): Le nombre de classes pour la classification (par défaut: 51).
        """
        super(FlauBertClass, self).__init__()
        self.flaubert_model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased")
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attn_mask):
        outputs = self.flaubert_model(input_ids, attention_mask=attn_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        pooler = self.pre_classifier(hidden_state)
        pooler_probabilities = torch.sigmoid(pooler)
        output_dropout = self.dropout(pooler_probabilities)
        output = self.linear(output_dropout)
        return output



class PredictionModele():
    def __init__(self,path='BestModel/modelFlauBert.pth') -> None:
        
        """
        Description : Classe pour effectuer des prédictions avec le modèle BERT .

        Args:
            
            path (str)         : Chemin vers le modèle entraîné.
        """
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model=FlauBertClass()
        model.eval()
        model.load_state_dict(torch.load(path))
        model = model.to(self.device)
        
        
        self.model= model
        self.MAX_LEN = 256
        self.tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
       
        self.target_list=["building_works","studies_consulting_assistance","transport_and_related_services",
"financial_and_insurance_services","sewerage","car_industry_products_and_transport_equipment",
"real_estate_missions_and_inspections","it_services","cleaning_and_hygiene_services",
"building_equipment","medical_precision_optical_and_watchmaking_instruments",
"research_and_development","civil_engineering","green_spaces",
"printing_and_related_services","street_development","office_and_it_equipment",
"industrial_equipment_and_tools","electrical_equipment","chemical_products",
"structural_works","recreational_cultural_and_sports_services","electricity_gas_and_heat",
"food_beverages_and_related_products","health_and_social_work_services",
"education_and_training_services","radio_television_and_communication_equipment",
"clothing_and_textile_industry","roofing","communication_and_marketing_services",
"energy_and_environment","waste_management","ores_materials_and_construction_materials",
"software_supply","defence_and_security","petroleum_products_fuels",
"office_furniture_and_supplies","it_equipment_and_consumables","provision_of_meals",
"fire","postal_and_telecommunications_services","silvicultural_products",
"drinking_water","agriculture","wholesale_and_retail","water_related_works",
"mining_and_minerals","paper_and_cardboard","recreational_cultural_and_sports_equipment",
"rubber_or_plastic_products","hydraulic_equipment"]
    
       
    
 
    
    def get_prediction(self,texte:str):

        """
        Obtient les prédictions du modèle pour un texte donné.

        Args:
            texte (str): Texte à prédire.

        Returns:
            list: Liste des noms des classes (liste des catégories).
        """
        pred=[]
        
        encoded_text = self.tokenizer.encode_plus(
        texte,
        max_length=self.MAX_LEN,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        token_type_ids = encoded_text['token_type_ids'].to(self.device)
        output = self.model(input_ids, attention_mask)

        output = torch.sigmoid(output).detach().cpu()
        output_pre = output.flatten().round().numpy()

        # Correction si aucune catégorie n'est prédite (probabilité > seuil): choisir la catégorie ayant la plus grande probabilité comme la catégorie du texte
        if (output_pre.sum() == 0):
            
            max_label_index = output.argmax()
            output_pre[max_label_index] = 1  

        for idx, p in enumerate(output_pre):
            
            if p==1.:
                pred.append(self.target_list[idx])
        

        return pred
        
    
    
def main(texte="""recherche dune prestation dassistance pour mener 
                                  avec lassistance et les compétences techniques du fournisseur 
                                  les travaux relatifs « Prestation de déploiement au sein du MITI Deplog »
                                 """,path='BestModel/modelFlauBert.pth'):
    
    prediction=PredictionModele(path)
    pred=prediction.get_prediction(texte)
    print("Texte      : ",texte)
    print("Prediction : ",pred)
    return pred

    
if __name__ == "__main__":
    main()

     