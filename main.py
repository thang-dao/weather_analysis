import os
import logging
import pandas as pd
import statsmodels.api as sm
from functools import lru_cache
from dotenv import load_dotenv
from utils import read_json
load_dotenv(dotenv_path='config/.env')

logger = logging.getLogger('Weather-Prediction')

class WeatherPrediction:
    def __init__(self):
        self.base_input = read_json(os.getenv('BASE_INPUT'))
        self.model_dict = os.getenv('MODEL_DICT')
        self.models = dict()
        for k,v in self.model_dict:
            self.models[k] = self._load_model(v)
    
    def _load_model(self, model_inf):
        logger.info('Loading model: {}'.format(model_inf['name']))
        model = sm.load(model_inf['dir'])
        return model
        
    def preprocess(self, x):
        input_model = self.base_input.copy()
        for k,v in x.items():
            if k in input_model.keys():
                input_model[k] = v
            else:
                if k == 'Time':
                    input_model['Time_'+str(v)] = 1
                elif k == 'Month':
                    input_model['Month_'+str(v)] = 1
                elif k == 'Weather':
                    input_model['Weather_'+str(v)] = 1
        df_model = pd.DataFrame(input_model)
        return df_model
            
    @lru_cache(maxsize=16)
    def predict(self, inputs, model_type=0):
        model_input = self.preprocess(inputs)
        y = self.models[model_type].predict(model_input)
        pred = list(map(round, y))
        return pred