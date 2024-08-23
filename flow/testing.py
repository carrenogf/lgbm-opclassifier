import pandas as pd
import json
import joblib
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score  # Metricas
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from datetime import datetime

def ploimportance(importance_df,path):
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importancia de la característica')
    plt.ylabel('Características')
    plt.title('Importancia de las Características en el Modelo')
    plt.gca().invert_yaxis()
    plt.savefig(path)
    
def plotcm(cm,path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(path)


def test(path_params_base):
    try:
        params_base = json.load(open(path_params_base))
        df = pd.read_csv(params_base["DATASET_PROCESSED"],sep=";")
        params_work = json.load(open(params_base["PREPROCESSING_PARAMS"]))
        
        
        SEED = params_base["SEED"]
        TEST_SIZE = params_base["TEST_SIZE"]
        
        features = params_work["features_finales"]
        
        X = df[features]
        y = df.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        print("Cargando el modelo optimo")
        study = optuna.load_study(study_name=params_base["STUDY_NAME"], storage=params_base["BBDD_OPTUNA"])
        
        best_model = lgb.LGBMClassifier(**study.best_params)
        best_model.fit(X_train, y_train)
        joblib.dump(best_model, params_base["MODEL"])
        y_pred = best_model.predict(X_test)
        
        # Obtener la importancia de las características
        importance = best_model.feature_importances_

        # Crear un DataFrame para visualizar mejor
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        importance_df.to_csv(f'{params_base["PATH_MODELS"]}/importance_{params_base["STUDY_NAME"]}.csv',index=False,sep=";")
        

        
        # metricas
        test_kappa = cohen_kappa_score(y_test, y_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        with open(f'{params_base["PATH_MODELS"]}/metrics_{params_base["STUDY_NAME"]}.txt', 'w') as f:
            f.write(f'study_name: {params_base["STUDY_NAME"]}\n')
            f.write(f'Fecha y hora: {datetime.now()}\n')
            f.write(f'Kappa: {test_kappa}\n')
            f.write(f'Accuracy: {test_accuracy}\n')
            f.write(f'dimensióm Test: {X_test.shape}\n')
            

        # Crear la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plotcm(cm,f'{params_base["PATH_MODELS"]}/confusion_matrix_{params_base["STUDY_NAME"]}.png')
        ploimportance(importance_df,f'{params_base["PATH_MODELS"]}/importance_{params_base["STUDY_NAME"]}.png')

    except Exception as e:
        return str(e)
    
    
