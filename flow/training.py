import pandas as pd
import json
# Funciones auxiliares sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer  # Metricas
import lightgbm as lgb
import optuna

def train(path_params_base):
    try:
        params_base = json.load(open(path_params_base))
        df = pd.read_csv(params_base["DATASET_PROCESSED"],sep=";")
        params_work = json.load(open(params_base["PREPROCESSING_PARAMS"]))
        
        
        SEED = params_base["SEED"]
        TEST_SIZE = params_base["TEST_SIZE"]
        features = params_work["features_finales"]
        
        X = df[features]
        y = df.target
        n_features = len(set(y))
        # División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
        
        kappa_scorer = make_scorer(cohen_kappa_score)
        
        hiperParameters = params_base["HP"]
        
        def objective(trial):
        # Contruir los hiperparámetros a optimizar
            param_training = {}
            param_training['num_class'] = n_features

            for p in hiperParameters.keys():

                if isinstance(hiperParameters[p], str) or isinstance(hiperParameters[p], int) or isinstance(hiperParameters[p], float):
                    param_training[p] = hiperParameters[p]

                elif isinstance(hiperParameters[p], tuple) or isinstance(hiperParameters[p], list):
                    if p in params_base["PARAMS_INT"]:
                        param_training[p] = trial.suggest_int(p, hiperParameters[p][0], hiperParameters[p][1])
                    elif p in params_base["PARAMS_UNIFORM"]:
                        param_training[p] = trial.suggest_uniform(p, hiperParameters[p][0], hiperParameters[p][1])
                    elif p in params_base["PARAMS_LOGUNIFORM"]:
                        param_training[p] = trial.suggest_float(p, hiperParameters[p][0], hiperParameters[p][1],log=True)

            # Crear el dataset de LightGBM
            model = lgb.LGBMClassifier(**param_training,verbose_eval=False)

            # Realizar validación cruzada usando Kappa como métrica
            kappa = cross_val_score(model, X_train, y_train, cv=3, scoring=kappa_scorer).mean()

            return kappa

    
        # Crear un estudio y optimizar
        study = optuna.create_study(direction='maximize', 
                                    storage=params_base["BBDD_OPTUNA"],  # Specify the storage URL here.
                                    study_name=params_base["STUDY_NAME"],
                                    load_if_exists=True)
        study.optimize(objective, n_trials=params_base["N_TRIALS"])
        
        print("Optimización de Hiperparámetros finalizada")

    except Exception as e:
        return str(e)


