import os
import logging
import pandas as pd
import statsmodels.api as sm
from functools import lru_cache
from dotenv import load_dotenv
from utils import read_json
load_dotenv(dotenv_path='config/.env')

logger = logging.getLogger('Weather-Prediction')
RAINY_LABEL = ['Light drizzle','Light rain','Light rain shower', 'Patchy light drizzle', 'Patchy light rain',
             'Patchy light rain with thunder','Patchy rain possible','Heavy rain','Heavy rain at times',
             'Moderate or heavy rain shower','Moderate rain', 'Moderate rain at times', 'Overcast','Torrential rain shower']


class WeatherPrediction:
    def __init__(self):
        self.base_input = read_json(os.getenv('BASE_INPUT'))
        self.model_dict = read_json(os.getenv('MODEL_DICT'))
        self.class_name = read_json(os.getenv('CLASS_NAME'))
        self.models = dict()
        for model_inf in self.model_dict:
            model_ins, name = self._load_model(model_inf)
            self.models[name] = model_ins
             
    def _load_model(self, model_inf):
        logger.info('Loading model: {}'.format(model_inf['name']))
        model = sm.load(model_inf['dir'])
        return model, model_inf['type']

    def preprocess(self, x):
        if x['Weather'] in RAINY_LABEL:
            x['Weather'] = 'rainy'
        else:
            x['Weather'] = 'sunny'
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
        df_model = pd.DataFrame([input_model])
        return df_model

    def predict(self, inputs, model_type=0):
        model_input = self.preprocess(inputs)
        assert model_input.shape == (1,33)
        y = self.models[model_type].predict(model_input)
        pred = list(map(round, y))
        return self.class_name[str(pred[0])]
    
    
if __name__=='__main__':
    predictor = WeatherPrediction()
    test_sample = read_json('config/test_sample.json')
    pred = predictor.predict(test_sample)
    print(pred)