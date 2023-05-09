import sys
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainerConfig, ModelTrainer



@dataclass
class DataIngestioConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')



class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestioConfig()

    def intiate_data_ingestion(self):
        logging.info("Data ingestion has started")
        try:
            '''
            code fopr fetching data 
            '''
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Read the data as Dataframe')
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=41)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True) 

            logging.info("Ingestion of the data is compleated")

            return (
                self.data_ingestion_config.test_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data = obj.intiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

