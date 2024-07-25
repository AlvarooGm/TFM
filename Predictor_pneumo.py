
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight, resample
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

# Verifica si estás usando GPU
print("Num GPUs encontradas: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.gpu_device_name():
    print(f"GPU: {tf.test.gpu_device_name()}")
else:
    print("GPU no encontrada.")

# Configuración de generadores de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Directorios de datos
train_dir = 'data_/train'
val_dir = 'data_/val'
test_dir = 'data_/test'

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=50,  # Ajusta según tu hardware
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=50,  # Ajusta según tu hardware
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=50,  # Ajusta según tu hardware
    class_mode='binary'
)

# Ajuste del peso de las clases basado en la proporción del dataset
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Definición del modelo
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
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

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)


# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    class_weight=class_weights,  # Se aplica el ajuste de peso de clases
    callbacks=[early_stopping, model_checkpoint]
)

# Función para graficar el historial del entrenamiento
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

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

# Evaluación del modelo
# Función para calcular y mostrar las métricas de evaluación adicionales
def evaluar_modelo(generator, model):
    # Obtener las etiquetas verdaderas y las predicciones del modelo
    etiquetas = generator.classes
    predicciones = model.predict(generator)
    predicciones_binarias = (predicciones > 0.5).astype(int).flatten()

    # Calcular precisión, sensibilidad, especificidad y AUC-ROC
    precision = precision_score(etiquetas, predicciones_binarias)
    sensibilidad = recall_score(etiquetas, predicciones_binarias)
    fpr, tpr, _ = roc_curve(etiquetas, predicciones)
    auc_roc = roc_auc_score(etiquetas, predicciones)
    especificidad = (1 - fpr[1])  # Specificity calculation
    # Imprimir las métricas
    print(f'Precisión: {precision*100:.2f}%')
    print(f'Sensibilidad: {sensibilidad*100:.2f}%')
    print(f'Especificidad: {especificidad*100:.2f}%')
    print(f'AUC-ROC: {auc_roc:.2f}')

   
# Evaluar el modelo
evaluar_modelo(test_generator, model)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {accuracy*100:.2f}%')

# Función para graficar la imagen con la predicción
def graficar_imagen(i, arr_predicciones, imagenes):
    prediccion = arr_predicciones[i][0]
    img = imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    porcentaje_covid = prediccion * 100
    porcentaje_normal = (1 - prediccion) * 100

    plt.xlabel(f"COVID: {porcentaje_covid:.2f}%\nNormal: {porcentaje_normal:.2f}%")

# Función para graficar el valor del arreglo de predicciones
def graficar_valor_arreglo(i, arr_predicciones):
    prediccion = arr_predicciones[i][0]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar([0, 1], [1 - prediccion, prediccion], color=["blue", "red"])
    plt.ylim([0, 1])

    plt.xticks([0, 1], ['Normal', 'COVID'])
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
