import torch

class PreparationDesDonnees(torch.utils.data.Dataset):
    
    """
        Classe pour préparer les données en vue de les utiliser dans un DataLoader.

        Args:
            df (DataFrame)        : DataFrame contenant les données.
            tokenizer (Tokenizer) : Tokenizer pour le texte.
            max_len (int)         : Longueur maximale des séquences.
            target_list (list)    : Liste des noms des classes (liste des catégories).
    """
    
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.texte = list(df['Text'])
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.texte)

    def __getitem__(self, index):
        texte = str(self.texte[index])
        texte = " ".join(texte.split())
        inputs = self.tokenizer.encode_plus(
            texte,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'texte': texte
        }