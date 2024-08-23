import joblib
import pandas as pd
import json


def predict(path_params_base, dataset_final):
    try:
        params_base = json.load(open(path_params_base))
        df = pd.read_csv(params_base["DATASET_PREDICT_PROCESSED"],sep=";")
        df_original = pd.read_csv(params_base["DATASET_PREDICT"],sep=";")
        params_preprocessing = json.load(open(params_base["PREPROCESSING_PARAMS"]))
        df = df[params_preprocessing["features_finales"]]
        
        model = joblib.load(params_base["MODEL"])
        
        y_pred = model.predict(df)

        print(f'modelo {params_base["STUDY_NAME"]} cargado')
        df_original["predict"] = y_pred
        
        clases = params_preprocessing["clases"]
        keys = list(clases.keys())
        def get_categoria(value):
            return keys[value]
        
        
        df_original.loc[df_original["clase"].isna(),'clase'] = df_original["predict"].apply(get_categoria)
        
        df_original.to_csv(dataset_final, index=False,sep=";")
        print("Proceso Terminado")
    except Exception as e:
        return str(e)
