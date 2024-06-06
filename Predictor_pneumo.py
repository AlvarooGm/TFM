import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Reescalada de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Generamos los datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    'TFM/data_/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Generamos los datos de validación
validation_generator = val_test_datagen.flow_from_directory(
    'TFM/data_/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# Generamos los  datos de prueba
test_generator = val_test_datagen.flow_from_directory(
    'TFM/data_/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
# Función para graficar las curvas de pérdida y precisión
def graficar_historial(history):
    # Curvas de pérdida
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Curvas de Pérdida')

    # Curvas de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Curvas de Precisión')

    plt.show()

# Graficar el historial del entrenamiento
graficar_historial(history)


loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {accuracy*100:.2f}%')


# Función para graficar la imagen con la predicción
def graficar_imagen(i, arr_predicciones, imagenes):
    prediccion = arr_predicciones[i][0]  # Asegurarse de tomar el primer valor si es un array
    img = imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)
    
'''  porcentaje_pneumonia = prediccion * 100
    porcentaje_normal = (1 - prediccion) * 100

    plt.xlabel(f"Pneumonia: {porcentaje_pneumonia:.2f}%\nNormal: {porcentaje_normal:.2f}%")'''

# Función para graficar el valor del arreglo de predicciones
def graficar_valor_arreglo(i, arr_predicciones):
    prediccion = arr_predicciones[i][0]  # Asegurarse de tomar el primer valor si es un array
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar([0, 1], [1 - prediccion, prediccion], color=["red", "blue"])
    plt.ylim([0, 1])

    # Etiquetas para las barras
    plt.xticks([0, 1], ['Normal', 'Pneumonia'])
    for bar, percentage in zip(grafica, [1 - prediccion, prediccion]):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{percentage*100:.2f}%', va='bottom') 

# Obtener un lote de datos de prueba
imagenes_prueba, _ = next(test_generator)
predicciones = model.predict(imagenes_prueba)

# Parámetros de la cuadrícula
filas = 5
columnas = 5
num_imagenes = filas * columnas

# Crear la figura para las imágenes y predicciones
plt.figure(figsize=(2 * 2 * columnas, 2 * filas))

for i in range(num_imagenes):
    plt.subplot(filas, 2 * columnas, 2 * i + 1)
    graficar_imagen(i, predicciones, imagenes_prueba)
    plt.subplot(filas, 2 * columnas, 2 * i + 2)
    graficar_valor_arreglo(i, predicciones)

plt.tight_layout()
plt.show()