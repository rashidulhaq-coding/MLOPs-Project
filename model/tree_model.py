from .base_model import BaseModel
from dataloader.dataloader import DataLoader 
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk 

class TreeModel(BaseModel):
    def __init__(self,config):
        super().__init__(config)

        self.base_model = RandomForestClassifier(**self.config.model.params)
        