import tensorflow as tf
import numpy as np

# Datos de entrenamiento 
turbina_potencia = np.array([100, 200, 150, 120, 180])  # Potencia de la turbina en Watts
tanque_capacidad = np.array([500, 800, 600, 700, 900])  # Capacidad del tanque en litros
tiempo_llenado = np.array([10, 12, 9, 11, 14])  # Tiempo de llenado en minutos

# Normalizar los datos (escala entre 0 y 1)
turbina_potencia = turbina_potencia / 200
tanque_capacidad = tanque_capacidad / 900
tiempo_llenado = tiempo_llenado / 14

# Crear el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(np.column_stack((turbina_potencia, tanque_capacidad)), tiempo_llenado, epochs=1000)

# Hacer una predicción
turbina_potencia_input = 0.8  # Potencia de la turbina en Watts (debe estar normalizado)
tanque_capacidad_input = 0.7  # Capacidad del tanque en litros (debe estar normalizado)

predicted_time = model.predict(np.array([[turbina_potencia_input, tanque_capacidad_input]]))
predicted_time = predicted_time * 14  # Desnormalizar el tiempo

print(f"Tiempo estimado de llenado: {predicted_time[0][0]:.2f} minutos")
