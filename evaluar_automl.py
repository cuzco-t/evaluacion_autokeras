import gc
import logging
import os
import time
import traceback
import tempfile
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

# Asegúrate de que openml_descargador.py y result.py estén en el path
from openml_descargador import OpenMLDescargador

# ----------------------------------------------------------------------
# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluacion")

from dotenv import load_dotenv
load_dotenv() 

# ----------------------------------------------------------------------
# Configuración de GPU y Keras - ANTES de importar AutoKeras
import tensorflow as tf
import os

# Establecer backend de Keras a TensorFlow (importante para AutoKeras 3.0.0)
os.environ['KERAS_BACKEND'] = 'tensorflow'

# Configuración de GPU - Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger_temp = logging.getLogger("config_gpu")
        logger_temp.info(f"GPU configurada. Dispositivos encontrados: {len(gpus)}")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("No se detectaron dispositivos GPU")

# Importaciones opcionales de las herramientas AutoML
try:
    import autokeras as ak
    AUTOKERAS_DISPONIBLE = True
except ImportError as e:
    print(f"Error al importar AutoKeras: {e}")
    AUTOKERAS_DISPONIBLE = False
    print("AutoKeras no está instalado. Se omitirán sus ejecuciones.")

# ----------------------------------------------------------------------
# Conexión a PostgreSQL
def obtener_conexion():
    """Devuelve una conexión a la base de datos usando variables de entorno."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        logger.error(f"No se pudo conectar a PostgreSQL: {e}")
        raise

def asegurar_conexion(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
    except:
        conn = obtener_conexion()
    return conn

# ----------------------------------------------------------------------
# Inserción de resultados
INSERT_QUERY = sql.SQL("""
    INSERT INTO otros_automl (
        nombre_automl, task_id, nombre_dataset, fuente, tiempo,
        f1, accuracy, "precision", recall,
        mae, mse, rmse, medae, ev, r2,
        silhouette, calinski, davies
    ) VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        NULL, NULL, NULL
    )
""")

def guardar_resultado(conn, registro: dict):
    """Inserta un registro en la tabla otros_automl."""
    with conn.cursor() as cur:
        cur.execute(INSERT_QUERY, (
            registro["nombre_automl"],
            registro["task_id"],
            registro["nombre_dataset"],
            registro["fuente"],
            registro["tiempo"],
            registro["f1"],
            registro["accuracy"],
            registro["precision"],
            registro["recall"],
            registro["mae"],
            registro["mse"],
            registro["rmse"],
            registro["medae"],
            registro["ev"],
            registro["r2"],
        ))

# ----------------------------------------------------------------------
# Preparación de los datos
def cargar_y_dividir(task_id: int, tipo: str):
    """
    Descarga dataset, divide en train/test.
    Retorna (X_train, X_test, y_train, y_test, nombre_dataset) o lanza excepción.
    """
    descargador = OpenMLDescargador()
    resultado = descargador.obtener_datos_tarea(task_id)
    if not resultado.is_success:
        raise RuntimeError(f"Fallo en descarga: {resultado.get_error()}")

    nombre, _, X, y = resultado.unwrap()

    # Dividir (20% test)
    stratify = y if tipo == "clasificacion" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, nombre

# ----------------------------------------------------------------------
# Funciones de evaluación por herramienta
def _preparar_datos_para_autokeras(tipo, X_train, y_train, X_test, y_test):
    """Convierte las variables a los tipos exactos que AutoKeras espera."""

    # Convertir y a pandas Series con el dtype correcto
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    if tipo == "clasificacion":
        # Forzar a entero de Python (categorías 0..n-1)
        y_train = y_train.astype('category').cat.codes
        y_test = y_test.astype('category').cat.codes
        # Si hay una sola clase, añadir una artificial (la omitiremos más tarde)
        if y_train.nunique() < 2:
            logger.warning(f"Task actual: Solo una clase. AutoKeras no puede clasificar.")
            return None, None, None, None, True   # señal para omitir
    else:
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

    # Convertir X a DataFrame con tipos simples
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Reemplazar tipos objet por string para que AutoKeras los trate como categóricos
    for col in X_train.columns:
        if X_train[col].dtype == object:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy(), False

def evaluar_autokeras(tipo, X_train, y_train, X_test, y_test, task_id):
    temp_dir = None
    try:
        X_tr, y_tr, X_te, y_te, omitir = _preparar_datos_para_autokeras(
            tipo, X_train, y_train, X_test, y_test
        )
        if omitir:
            return None, 0.0

        # Crear carpeta temporal para AutoKeras (se elimina automáticamente después)
        temp_dir = tempfile.mkdtemp(prefix="autokeras_")

        if tipo == "clasificacion":
            modelo = ak.StructuredDataClassifier(
                overwrite=True, directory=temp_dir, seed=912, max_trials=15
            )
        else:
            modelo = ak.StructuredDataRegressor(
                overwrite=True, directory=temp_dir, seed=912, max_trials=15
            )

        inicio = time.perf_counter()
        modelo.fit(X_tr, y_tr, epochs=100)
        y_pred = modelo.predict(X_te)

        tiempo = time.perf_counter() - inicio

        if tipo == "clasificacion":
            if y_pred.ndim > 1:
                y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = y_pred.flatten()

        metricas = calcular_metricas(tipo, y_te, y_pred)
        return metricas, tiempo

    except Exception as e:
        logger.error(f"AutoKeras task {task_id}: {e}\n{traceback.format_exc()}")
        return None, 0.0
    finally:
        # Limpiar carpeta temporal
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"No se pudo eliminar carpeta temporal {temp_dir}: {e}")

        try:
            tf.keras.backend.clear_session()
        except:
            pass

        # eliminar referencias grandes explícitamente
        try:
            del modelo
        except:
            pass

        gc.collect()

# ----------------------------------------------------------------------
# Cálculo de métricas
def calcular_metricas(tipo: str, y_true, y_pred):
    """Calcula todas las métricas necesarias. Las no aplicables se ponen en None."""
    met = {
        "f1": None, "accuracy": None, "precision": None, "recall": None,
        "mae": None, "mse": None, "rmse": None, "medae": None,
        "ev": None, "r2": None,
    }
    if tipo == "clasificacion":
        # Weighted para soportar múltiples clases
        met["accuracy"] = float(accuracy_score(y_true, y_pred))
        met["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        met["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        met["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    else:  # regresion
        met["mae"] = float(mean_absolute_error(y_true, y_pred))
        met["mse"] = float(mean_squared_error(y_true, y_pred))
        met["rmse"] = float(np.sqrt(met["mse"]))
        met["medae"] = float(median_absolute_error(y_true, y_pred))
        met["ev"] = float(explained_variance_score(y_true, y_pred))
        met["r2"] = float(r2_score(y_true, y_pred))
    return met

def get_metricas_error():
    """Retorna un diccionario de métricas con valor -1111 para indicar error."""
    return {
        "f1": -1111,
        "accuracy": -1111,
        "precision": -1111,
        "recall": -1111,
        "mae": -1111,
        "mse": -1111,
        "rmse": -1111,
        "medae": -1111,
        "ev": -1111,
        "r2": -1111,
    }

# ----------------------------------------------------------------------
# Procesamiento de un archivo
def procesar_archivo(ruta: str, tipo: str, conn):
    """
    Lee task_ids de 'ruta' y ejecuta las dos herramientas.
    """
    fuente = os.path.splitext(os.path.basename(ruta))[0]
    with open(ruta, "r") as f:
        lineas = [line.strip() for line in f if line.strip()]

    for task_id_str in lineas:
        try:
            task_id = int(task_id_str)
        except ValueError:
            logger.warning(f"Línea no numérica ignorada en {ruta}: {task_id_str}")
            continue

        logger.info(f"Iniciando task_id={task_id} ({tipo})")
        X_train, X_test, y_train, y_test, nombre_dataset = None, None, None, None, ""

        try:
            X_train, X_test, y_train, y_test, nombre_dataset = cargar_y_dividir(task_id, tipo)
        except Exception as e:
            logger.error(f"Error al cargar datos del task {task_id}: {e}\n{traceback.format_exc()}")
            continue

        # Probar AutoKeras
        if AUTOKERAS_DISPONIBLE:
            try:
                metricas, tiempo = evaluar_autokeras(tipo, X_train, y_train, X_test, y_test, task_id)
                registro = {
                    "nombre_automl": "autokeras",
                    "task_id": task_id,
                    "nombre_dataset": nombre_dataset,
                    "fuente": fuente,
                    "tiempo": round(tiempo, 2),
                    **metricas,
                }
                try:
                    conn = asegurar_conexion(conn)
                    guardar_resultado(conn, registro)
                    logger.info(f"Guardado OK: autokeras, task {task_id}")
                except Exception as e:
                    logger.error(f"Error al insertar en DB (autokeras, task {task_id}): {e}")
            except Exception as e:
                logger.error(f"Fallo en AutoKeras para task {task_id}: {e}\n{traceback.format_exc()}")
                # Registrar en BD con métricas de error (-1111)
                metricas_error = get_metricas_error()
                registro_error = {
                    "nombre_automl": "autokeras",
                    "task_id": task_id,
                    "nombre_dataset": nombre_dataset,
                    "fuente": fuente,
                    "tiempo": 0,  # Sin tiempo ya que falló
                    **metricas_error,
                }
                try:
                    conn = asegurar_conexion(conn)
                    guardar_resultado(conn, registro_error)
                    logger.info(f"Guardado REGISTRO DE ERROR: autokeras, task {task_id} con métricas=-1111")
                except Exception as db_error:
                    logger.error(f"Error al insertar registro de error en DB (autokeras, task {task_id}): {db_error}")
        else:
            logger.info(f"AutoKeras no disponible, se omite task {task_id}")


# ----------------------------------------------------------------------
# Punto de entrada
def main():
    archivos = [
        ("data/openml-cc18.txt", "clasificacion"),
        ("data/openml-ctr23.txt", "regresion"),
    ]

    conn = None
    try:
        conn = obtener_conexion()
        logger.info("Conexión a base de datos establecida.")
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        logger.critical("No se puede continuar sin base de datos.")
        return

    try:
        for archivo, tipo in archivos:
            if not os.path.exists(archivo):
                logger.warning(f"Archivo no encontrado: {archivo}, se omite.")
                continue
            logger.info(f"Procesando archivo {archivo} ({tipo})")
            try:
                procesar_archivo(archivo, tipo, conn)
            except Exception as e:
                logger.critical(f"Error inesperado en archivo {archivo}: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Conexión a base de datos cerrada.")

if __name__ == "__main__":
    main()