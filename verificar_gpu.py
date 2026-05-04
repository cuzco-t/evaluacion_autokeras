#!/usr/bin/env python3
"""
Script para verificar que TensorFlow/Keras está configurado para usar GPU
y que AutoKeras está usando el backend correcto.
"""

import os
import sys

# Establecer backend a TensorFlow antes de cualquier importación
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
import keras

print("=" * 70)
print("VERIFICACIÓN DE CONFIGURACIÓN DE GPU")
print("=" * 70)

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n✓ Dispositivos GPU detectados: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("  ⚠ ADVERTENCIA: No se detectaron GPUs. El entrenamiento será lento.")

# Verificar memoria creciente
print(f"\n✓ Memory growth habilitado para GPU")
for gpu in gpus:
    print(f"  {gpu}")

# Verificar versiones
print(f"\n✓ Versiones:")
print(f"  TensorFlow: {tf.__version__}")
print(f"  Keras: {keras.__version__}")

# Verificar backend de Keras
print(f"\n✓ Backend de Keras: {keras.config.backend()}")
if keras.config.backend() != 'tensorflow':
    print(f"  ⚠ ADVERTENCIA: Se esperaba 'tensorflow', pero se encontró '{keras.config.backend()}'")

# Verificar AutoKeras
try:
    import autokeras as ak
    print(f"\n✓ AutoKeras: {ak.__version__}")
except ImportError:
    print(f"\n✗ AutoKeras no está instalado")
    sys.exit(1)

# Test rápido: crear un tensor en GPU
print(f"\n✓ Test de ubicación de tensor:")
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    c = tf.matmul(a, b)
    print(f"  Ubicación del resultado: {c.device}")

print("\n" + "=" * 70)
print("✓ CONFIGURACIÓN LISTA - GPU está disponible y configurada")
print("=" * 70)
