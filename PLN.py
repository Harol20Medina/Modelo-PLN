# Importación de librerías necesarias
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Crear un conjunto de datos de ejemplo con frases y etiquetas simuladas
data = {
    'text': [
        "El equipo ganó el campeonato", "Nuevo lanzamiento de smartphone", "El presidente anuncia nuevas políticas",
        "Gran victoria en el partido de ayer", "La compañía presentó su nuevo dispositivo", "Discusión sobre la reforma educativa",
        "Famoso jugador ficha por otro club", "La actualización mejora la seguridad", "Debate sobre la economía nacional",
        "Se esperan grandes fichajes en la temporada", "Nueva tecnología permite cargar más rápido", "Propuesta de ley en el parlamento",
        "Final emocionante en el torneo", "El mercado de teléfonos sigue creciendo", "Elecciones nacionales programadas para abril",
        "Jugador estrella marca tres goles", "Análisis de la última actualización de software", "Nuevas alianzas políticas",
        "El entrenador confía en el equipo", "Anuncian mejoras en los procesadores", "Protestas en las calles de la capital",
        "Especulaciones sobre el futuro del entrenador", "La industria de videojuegos crece rápidamente", "Reformas en el sistema de salud",
        "Partido intenso entre los equipos locales", "Nuevo sistema operativo lanzado", "Propuestas para mejorar la educación",
        "Fútbol: el equipo se prepara para el clásico", "El impacto de la inteligencia artificial", "Plan de desarrollo económico aprobado",
        "Jugadores entrenan para la final", "La empresa lanza un gadget innovador", "Discusión sobre políticas ambientales",
        "Gran actuación en el campeonato", "Nuevo chip promete mayor rendimiento", "Se propone reducir los impuestos",
        "Fans celebran la victoria en el estadio", "La aplicación ahora es compatible con más dispositivos", "Crisis política en el gobierno",
        "Clasificación para el mundial comienza", "Desarrollan robot para tareas domésticas", "Revisión del presupuesto nacional",
        "Análisis de la liga europea", "Presentan nuevo vehículo autónomo", "Proyecto de ley ambiental presentado",
        "Jugador de baloncesto bate récords", "Novedades en la industria de semiconductores", "Discusión sobre derechos humanos",
        "Atletas se preparan para las olimpiadas", "La compañía invierte en energías renovables", "Propuestas de reforma laboral",
        "El equipo nacional vence a su rival", "El avance de la realidad aumentada", "Cumbre internacional sobre cambio climático",
        "Clasificación de la liga sigue en disputa", "Lanzamiento de app de realidad virtual", "Manifestación por derechos civiles",
        "Gol de último minuto da la victoria", "Análisis del mercado de smartphones", "Nuevas regulaciones en el parlamento",
        "Victoria contundente en la copa", "Apple lanza nueva versión de iOS", "Cambios en el gabinete ministerial",
        "Torneo de tenis reúne a los mejores", "Samsung anuncia nuevo dispositivo", "Reunión sobre políticas de inmigración",
        "El entrenador anuncia su retiro", "Innovaciones en el sector de la tecnología", "Nueva ley aprobada por el congreso",
        "Equipo femenino gana el campeonato", "Tecnología de 5G expande su alcance", "Debate sobre seguridad pública",
        "Clasificación de la liga española", "Nueva función en la app de mensajería", "Se aprueba proyecto de ley fiscal",
        "Jugador gana el premio al mejor del año", "Avances en inteligencia artificial", "Alianza entre partidos políticos",
        "Tecnología para la mejora de la educación", "Estrategias contra el cambio climático", "Progreso en la inteligencia artificial aplicada",
        "Desarrollo de nuevas vacunas", "Reformas políticas en Europa", "La inteligencia artificial en la seguridad pública",
        "Avances en la robótica", "El mercado de la música digital", "El crecimiento de las energías renovables"
    ] * 5  # Repetimos las frases para tener un conjunto de datos más grande
}

# Generamos etiquetas alternadas de deportes y política
data['label'] = ["deportes", "política"] * (len(data['text']) // 3)
data['label'] = data['label'][:len(data['text'])]  # Aseguramos que las etiquetas y textos coincidan

# Creamos un DataFrame con los textos y etiquetas
df = pd.DataFrame(data)

# Extraemos los textos y etiquetas en variables separadas
texts = df['text'].values
labels = df['label'].values

# Binarizamos las etiquetas para convertirlas en formato compatible con la red neuronal
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Configuración de los parámetros de procesamiento de texto y red neuronal
vocab_size = 1000       # Tamaño máximo del vocabulario a considerar
embedding_dim = 128     # Dimensión de los embeddings
max_length = 20         # Longitud máxima de secuencias
trunc_type = 'post'     # Forma de truncar las secuencias largas
padding_type = 'post'   # Forma de rellenar secuencias cortas

# Inicializamos el tokenizador y lo ajustamos a los textos de entrenamiento
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convertimos los textos en secuencias numéricas y los rellenamos o truncan a max_length
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Construimos el modelo de red neuronal
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),  # Capa de embeddings
    SpatialDropout1D(0.2),                                          # Dropout para regularización
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),                   # Capa LSTM para captar relaciones en secuencias
    Dense(32, activation='relu'),                                   # Capa densa con función de activación ReLU
    Dropout(0.5),                                                   # Dropout adicional para evitar overfitting
    Dense(len(label_binarizer.classes_), activation='softmax')      # Capa de salida con softmax para clasificación multiclase
])

# Compilamos el modelo especificando pérdida, optimizador y métrica
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamos el modelo con early stopping para evitar overfitting si la pérdida no mejora
history = model.fit(
    X_train_padded, y_train,
    epochs=10,
    validation_data=(X_test_padded, y_test),
    batch_size=16,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
)

# Evaluamos el modelo en el conjunto de prueba y mostramos pérdida y precisión
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=2)
print(f'Pérdida en el conjunto de prueba: {loss}')
print(f'Precisión en el conjunto de prueba: {accuracy}')

# Predicción y evaluación del modelo
y_pred = model.predict(X_test_padded)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convertimos probabilidades a etiquetas predichas
y_true_classes = np.argmax(y_test, axis=1)  # Convertimos etiquetas verdaderas a clase numérica
print(classification_report(y_true_classes, y_pred_classes, target_names=label_binarizer.classes_))

# Gráficas de la pérdida y precisión durante el entrenamiento
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento y validación')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión durante el entrenamiento y validación')
plt.legend()

plt.show()
