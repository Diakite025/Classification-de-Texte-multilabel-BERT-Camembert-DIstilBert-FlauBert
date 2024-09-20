import torch 
from transformers import BertTokenizer,BertModel,CamembertTokenizer,CamembertModel,AutoTokenizer, DistilBertModel,FlaubertModel

from pymongo import MongoClient
from gridfs import GridFSBucket

import os


class PredictionModel():

    #FlauBertClass
    class FlauBertClass(torch.nn.Module):
        def __init__(self, num_classes=51):
          
            super(PredictionModel.FlauBertClass, self).__init__()
            self.flaubert_model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased")
            self.pre_classifier = torch.nn.Linear(768, 768)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, num_classes)
        
        def forward(self, input_ids, attn_mask,token_type_ids):
            outputs = self.flaubert_model(input_ids, attention_mask=attn_mask)
            hidden_state = outputs.last_hidden_state[:, 0]
            pooler = self.pre_classifier(hidden_state)
            pooler_probabilities = torch.sigmoid(pooler)
            output_dropout = self.dropout(pooler_probabilities)
            output = self.linear(output_dropout)
            return output
        
        
    #DistilBertClass
    class DistilBertClass(torch.nn.Module):
    
   
        def __init__(self,num_classes=51):
            super(PredictionModel.DistilBertClass, self).__init__()
            self.DistilBert =DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.pre_classifier = torch.nn.Linear(768, 768)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, num_classes)

        def forward(self, input_ids, attn_mask,token_type_ids):
            output = self.DistilBert(input_ids, attention_mask=attn_mask)
            
            hidden_state = output.last_hidden_state[:, 0]
            pooler=self.pre_classifier(hidden_state)
            pooler_probabilities=torch.sigmoid(pooler)
            output_dropout = self.dropout(pooler_probabilities)
            output = self.linear(output_dropout)
            return output
        
    
    #CamemBERTClass
    class CamemBERTClass(torch.nn.Module):
        def __init__(self,num_classes=51):
            super(PredictionModel.CamemBERTClass, self).__init__()
            self.camembert_model =CamembertModel.from_pretrained("camembert-base", return_dict=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, num_classes)



        def forward(self, input_ids, attn_mask, token_type_ids):
            output = self.camembert_model(
                input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids
            )
            output_dropout = self.dropout(output.pooler_output)
            output = self.linear(output_dropout)
            return output

    #BertClass
    class BertClass(torch.nn.Module):

        def __init__(self,num_classes=51):
            super(PredictionModel.BertClass, self).__init__()
            self.bert_model =BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(768, num_classes)



        def forward(self, input_ids, attn_mask, token_type_ids):
            output = self.bert_model(
                input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids
            )
            output_dropout = self.dropout(output.pooler_output)
            output = self.linear(output_dropout)
            return output




    def __init__(self,db,model,language_id,model_type) -> None:


        grid = GridFSBucket(db,'models')

        file = db.models.files.find_one({"model":model, "language_id":language_id,"model_type":model_type})
        if (file ==None):
            raise Exception("model not found")

        path = "./models/"
        if not os.path.exists(path):
            os.makedirs(path)

        path = path + file["filename"]+"_"+file["model_type"]
        if not os.path.exists(path):
            fdw = open(path,"wb")
            grid.download_to_stream(file["_id"], fdw)
            fdw.close()


        if file["model_type"]=="camembert":
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            self.modele=self.CamemBERTClass()

        elif file["model_type"] == "bert":

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.modele=self.BertClass()
            
        elif file["model_type"] == "distilbert":

            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.modele=self.DistilBertClass()
            
        elif file["model_type"] == "flaubert":

            self.tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
            self.modele=self.FlauBertClass()

        else:
            raise Exception(" None model_type: "+file["model_type"])




        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.modele.eval()
        self.modele.load_state_dict(torch.load(path, map_location=self.device))
        self.modele =  self.modele.to(self.device)
        self.MAX_LEN = 256




        self.target_list= ["building_works","studies_consulting_assistance","transport_and_related_services",
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
        output = self.modele(input_ids, attention_mask, token_type_ids)

        output = torch.sigmoid(output).detach().cpu()
        output_pre = output.flatten().round().numpy()

        if (output_pre.sum() == 0):

            max_label_index = output.argmax()
            output_pre[max_label_index] = 1

        for idx, p in enumerate(output_pre):

            if p==1.:
                pred.append(self.target_list[idx])


        return pred



    
    
    
if __name__ == "__main__":
   mongo = MongoClient(lien)
   db = mongo.machine_learning
   prediction=PredictionModel(db,"category","fr","flaubert")
   pred=prediction.get_prediction("Services d'assistance technique Services d'assistance technique Services d'ingénierie")
   print(pred)




