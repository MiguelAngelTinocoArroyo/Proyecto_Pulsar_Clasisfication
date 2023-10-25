import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./Pulsar.csv')

# Función distancia entre cuartiles
def dist_inter_cuartil(df, variable, distancia):
    
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    limite_inf = df[variable].quantile(0.50) - (IQR * distancia)
    limite_sup = df[variable].quantile(0.75) + (IQR * distancia)
    return limite_sup, limite_inf

# Encontremos los límites superior e inferior para la variable 'SD' 

age_limite_sup, age_limite_inf = dist_inter_cuartil( df ,'SD' , 1.5)

# Reemplazando los valores extremos de la variabe 'SD' por 
# los límites máximos y mínimos

df['SD'] = np.where(df['SD'] > age_limite_sup, age_limite_sup,
                 np.where(df['SD'] < age_limite_inf, age_limite_inf, df['SD']))

# Eliminando las variables "EK", "Mean_DMSNR_Curve" por colinealidad
del df['EK']
del df['Mean_DMSNR_Curve']
del df['Skewness']
del df['EK_DMSNR_Curve']

# Separando la variables predictoras y la variable predictiva 
X = df.drop(['Class'], axis=1)
y = df['Class']

# Partición de Datos: Train, Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,
                                                    test_size = 0.3, random_state = 100)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, 
                            max_depth = 4,
                            criterion = 'gini', 
                            min_samples_leaf = 1,   # número mínimo de muestras requeridas para un nodo hoja válido
                            min_samples_split = 9,  # número mínimo de muestras requeridas para dividir un nodo no hoja
                            max_features = 3,
                            bootstrap = True,       # muestreo aleatorio
                            oob_score = True,
                            n_jobs = 5,
                            random_state = 100)
rf.fit(X_train, y_train)

# Entrenando el algoritmo para validar
y_predict_train = rf.predict(X_train)   # Predicción sobre el train
y_predict_test  = rf.predict(X_test)    # Predicción sobre el test 

# Hacer un archivo pickle de nuestro modelo
import pickle
pickle.dump(rf, open("model.pkl", "wb"))