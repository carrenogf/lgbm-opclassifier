import pandas as pd
import json
import sys
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Leer datos -----------------------------------------------------------------
# df1 son datos en local del 2023
def preprocess(path_params_base,path_train_local=None):
    
    try:
        nltk.download('stopwords')

        params_base = json.load(open(path_params_base))
        sys.path.append(params_base["PATH_FLOW"])
        import bbdd_caeto
        features_base = params_base["FEATURES_BASE"]

        # df2 son datos de 2024 den la base de datos de caeto
        table = params_base["TABLE"]
        columns = bbdd_caeto.getdata(f'select COLUMN_NAME from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = "{table}" ORDER BY ordinal_position;')
        df2 = pd.DataFrame(bbdd_caeto.getdata(f"select * from {table} WHERE codteso=1;"), columns=[column[0] for column in columns])
        print("Se cargan datos de las BBDD")
        
        if path_train_local:
            print("Se cargan datos locales")
            df1 = pd.read_excel(path_train_local)
            df1 = df1[features_base]
            df2 = df2[features_base]# se aplican los filtros para que coincidan las columnas
            df = pd.concat([df1,df2],axis=0) # se unen los datos
            df1 = None
            df2 = None
        # preproceso -----------------------------------------------------------------
        else:
            df2 = df2[features_base]
            df = df2.copy()
            df2 = None

        df.loc[df["clase_gasto"]=="   ",'clase_gasto'] = "" # limpieza del campo clase_gasto
        df["descripcion"] = df["descripcion"].fillna("")
        df["descripcion"] = df["descripcion"].str.replace(";","",regex=True)
        df = df[df["clase"].notna()] # se eliminan los registros sin clase
        df1 = None
        df2 = None

        # onehot encoding
        categ = params_base["FEATURES_CATEG"]
        for col in categ:
            df = pd.concat([df,pd.get_dummies(df[col],prefix=col, prefix_sep='_')],axis=1)
            df.drop(col, axis=1, inplace=True)
        print("Onehot encoding finalizado")
        # parametros de preprocesamiento
        # se creará un json con datos como el diccionario de clases, diccionario de palabras y lista de features completa
        params_preprocessing = {}

        # diccionario de clases y target ---------------------------------------------------
        Class = list(df.clase.unique())
        clases = {val:Class.index(val) for val in Class}
        def get_class(val):
            return clases[val]

        df['target'] = df['clase'].apply(get_class)

        params_preprocessing['clases'] = clases

        # proceso del texto de la glosa ----------------------------------------------------
        print("Procesando texto")
        # limpieza del texto
        def limpiar_texto(texto):
            texto = texto.upper()
            texto = re.findall(r"(?!CTA)(?!RES)(?!PAGO)(?!AGO)(?!PAG)[A-Z0-9]{3,}", texto)
            texto = " ".join(texto).strip()
            return texto

        df["texto_limpio"] = df["descripcion"].apply(limpiar_texto)

        def pesos(texto, dic_words):
            texto = texto.lower()
            palabras = texto.split(' ')
            score = 0
            for palabra in palabras:
                if palabra in dic_words.keys():
                    score += dic_words[palabra]
                    #print(palabra)
            return score

        # diccionario de palabras y calculo de pesos por categoria ---------------------------------------
        dictOfWords = {}
        for target in df.target.unique():
            df_target = df[df["target"]==target]
            all_descriptions = ' '.join(df_target['texto_limpio'].dropna())  # Concatenar todas las descripciones

            # Tokenización y eliminación de stopwords
            stop_words = set(stopwords.words('spanish')) 
            word_tokens = word_tokenize(all_descriptions.lower())  # Tokenización y convertir a minúsculas
            filtered_words = [word for word in word_tokens if word.isalnum() and word not in stop_words]  # Filtrar stopwords y no palabras alfa
            freq_of_words = pd.Series(filtered_words).value_counts()
            
            
            dic_words = freq_of_words.to_dict()
            dictOfWords[str(target)] = dic_words
            df[f'pesos_{target}'] = df['texto_limpio'].apply(pesos,dic_words=dic_words)

        params_preprocessing['dictOfWords'] = dictOfWords
        print("Procesamiento de texto finalizado")
        # Features finales -----------------------------------------------------------------
        features = []
        for x in enumerate(df.dtypes):
            if x[1] in ["float64","int64","bool"]:
                features.append(df.columns[x[0]])

        remove = [ 'target','nom_entidad1','nom_cta','glosa1','texto_limpio','descripcion']

        for col in remove:
            try:
                features.remove(col)
            except: pass

        params_preprocessing['features_finales'] = features

        # guardar parametros del proproceso

        with open(params_base["PREPROCESSING_PARAMS"], 'w') as file:
            json.dump(params_preprocessing, file, indent=4)
        print("Parametros de preprocesamiento guardados")
        # GUARDAR DATAFREMO PROCESADO
        df.to_csv(params_base["DATASET_PROCESSED"],index=False,sep=";")
        print("Dataframe preprocesado guardado")
        return "Preprocesamiento finalizado"

    except Exception as e:
        return str(e)
