import joblib
import sys
import json
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
def preprocess_predict(path_params_base):
    
    try:
        nltk.download('stopwords')

        params_base = json.load(open(path_params_base))
        sys.path.append(params_base["PATH_FLOW"])
        import bbdd_caeto
        
        params_preprocessing = json.load(open(params_base["PREPROCESSING_PARAMS"]))
        dictOfWords = params_preprocessing['dictOfWords']

        features_base = params_base["FEATURES_BASE"]
        print("Se cargan datos de las BBDD")
        # df2 son datos de 2024 den la base de datos de caeto
        table = params_base["TABLE_DEUDA"]
        columns = bbdd_caeto.getdata(f'select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = "{table}" ORDER BY ordinal_position;')
        df = pd.DataFrame(bbdd_caeto.getdata(f"select * from {table} WHERE codteso=1;"), columns=[column[0] for column in columns])
        
        # preproceso -----------------------------------------------------------------
        df.loc[df["clase_gasto"]=="   ",'clase_gasto'] = "" # limpieza del campo clase_gasto
        df["descripcion"] = df["descripcion"].fillna("")
        df["descripcion"] = df["descripcion"].str.replace(";","",regex=True)
        df_original = df.copy()
        df = df[features_base]
        # onehot encoding
        categ = params_base["FEATURES_CATEG"]
        for col in categ:
            df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
            df.drop(col, axis=1, inplace=True)
        print("Onehot encoding finalizado")

        # proceso del texto de la glosa ----------------------------------------------------
        print("Procesando texto")
        # limpieza del texto
        def limpiar_texto(texto):
            texto = texto.upper()
            texto = re.findall(r"(?!CTA)(?!RES)(?!PAGO)(?!AGO)(?!PAG)[A-Z0-9]{3,}", texto)
            texto = " ".join(texto).strip()
            return texto

        df["texto_limpio"] = df["descripcion"].apply(limpiar_texto)

        # diccionario de palabras y calculo de pesos por categoria ---------------------------------------
        def pesos(texto, dic_words):
            texto = texto.lower()
            palabras = texto.split(' ')
            score = 0
            for palabra in palabras:
                if palabra in dic_words.keys():
                    score += dic_words[palabra]
                    #print(palabra)
            return score

        for target in dictOfWords:
            dic_words = dictOfWords[target]
            df[f'pesos_{target}'] = df['texto_limpio'].apply(pesos,dic_words=dic_words)
            
            
        print("Procesamiento de texto finalizado")
        # Features finales -----------------------------------------------------------------
        for col in params_preprocessing['features_finales']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # GUARDAR DATAFREM PROCESADO
        df.to_csv(params_base["DATASET_PREDICT_PROCESSED"],index=False,sep=";")
        df_original.to_csv(params_base["DATASET_PREDICT"],index=False,sep=";")
        print("Dataframe procesado guardado")
        return "Preprocesamiento finalizado"

    except Exception as e:
        return str(e)
