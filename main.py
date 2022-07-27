import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)

RAINY_LABEL = ['Light drizzle','Light rain','Light rain shower', 'Patchy light drizzle', 'Patchy light rain',
            'Patchy light rain with thunder','Patchy rain possible','Heavy rain','Heavy rain at times',
            'Moderate or heavy rain shower','Moderate rain', 'Moderate rain at times', 'Overcast','Torrential rain shower']
RAINY_MONTH = [5,6,7,8,9,10,11]
MORNING_HOUR = [6,9,12,15]

def convert_category2numeric(data, columns_name):
    for col in columns_name:
        data = pd.concat((data, pd.get_dummies(data[col], prefix=col)), axis=1)
        data = data.drop([col], axis=1)
    return data

def prepare_data(file_path):
    df = pd.read_csv(file_path)

    label = df['Weather']
    features = df
    #remove data
    features = features[:-1]
    label = label[1:]
    
    data = features
    data['Label'] = label.values

    for i, value in enumerate(data.Label):
        if value in RAINY_LABEL:
            data.Label[i] = 'yes'
        else:
            data.Label[i] = 'no'

    for i, value in enumerate(data.Weather):
        if value in RAINY_LABEL:
            data.Weather[i] = 'rainy'
        else:
            data.Weather[i] = 'sunny'

    # remove columns
    # drop_columns = ['Day', 'Year', 'Humidity', 'Pressure']
    drop_columns = []
    data = data.drop(drop_columns, axis=1)

    data = convert_category2numeric(data, ['Time', 'Weather', 'Month'])
    data = pd.concat((data, pd.get_dummies(data['Label'], prefix='Label', drop_first=True)), axis=1)
    data = data.drop(['Label'], axis=1)

    all_columns = data.columns.to_list()
    Y_colums = ['Label_yes']
    X_colums = [x for x in all_columns if x not in Y_colums]
    
    return data, X_colums, Y_colums

def train(data_path):
    file_name = data_path.split('/')[-1][:-4]
    log_file = 'logs/{}.txt'.format(file_name)
    data, X_colums, Y_colums = prepare_data(data_path)
    train_set, test_set = train_test_split(data, test_size=0.2)
    Xtrain = train_set[X_colums]
    ytrain = train_set[Y_colums]

    # building the model and fitting the data
    log_reg = sm.Logit(ytrain, Xtrain).fit()
    print(log_reg.summary())

    Xtest = test_set[X_colums]
    ytest = test_set[Y_colums]

    yhat = log_reg.predict(Xtest)
    prediction = list(map(round, yhat))
    cm = confusion_matrix(ytest, prediction) 
    print ("Confusion Matrix : \n", cm)
    # accuracy score of the model
    print('Test accuracy = ', accuracy_score(ytest, prediction))
    with open(log_file, 'w') as fp:
        fp.write(str(log_reg.summary()))
        fp.write("\n")
        fp.write('Test accuracy = {}'.format(accuracy_score(ytest, prediction)))
    log_reg.save('models/{}.pickle'.format(file_name))
    return log_reg
    

def test(model_path, data_path):
    file_name = data_path.split('/')[-1][:-4]
    log_file = 'logs/test_{}.txt'.format(file_name)
    data, X_colums, Y_colums = prepare_data(data_path)
    model = sm.load(model_path)
    print(model.summary())
    Xtest = data[X_colums]
    ytest = data[Y_colums]
    yhat = model.predict(Xtest)
    prediction = list(map(round, yhat))
    cm = confusion_matrix(ytest, prediction) 
    print ("Confusion Matrix : \n", cm)
    # accuracy score of the model
    print('Test accuracy = ', accuracy_score(ytest, prediction))
    with open(log_file, 'w') as fp:
        fp.write(str(model.summary()))
        fp.write("\n")
        fp.write('Test accuracy = {}'.format(accuracy_score(ytest, prediction)))
    
    
if __name__=='__main__':
    # data_path = 'data/Bắc Bộ.csv'
    # model = train(data_path)
    # data_path = 'data/Nam Bộ.csv'
    # model = train(data_path)
    # data_path = 'data/Trung Bộ.csv'
    # model = train(data_path)

    data_path = 'data/Bà Rịa - Vũng Tàu.csv'
    model_path = 'models/Bắc Bộ.pickle'
    test(model_path, data_path)
    model_path = 'models/Trung Bộ.pickle'
    test(model_path, data_path)
    model_path = 'models/Nam Bộ.pickle'
    test(model_path, data_path)