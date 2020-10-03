import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

#Data Preparation Your CSV with labeled data
#title label
#abc   0
#xyz   1 
df = pd.read_csv('nlpdata.csv')

#Split data for training and validations
x_axis_train, x_axis_test, y_axis_train, y_axis_test = train_test_split(df['title'], 
                                                    df['label'], 
                                                    test_size=0.3, 
                                                    stratify=df['label'], 
                                                    random_state=40)

#Using TensorFlow hub to load sentence encoder
encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

#Creating Sequential model using keras api
model = tf.keras.models.Sequential()

#Transfer learning - Loading TF2.0 Saved Model sentence encoder
model.add(hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', 
                        input_shape=[], 
                        dtype=tf.string, 
                        trainable=True))

#adding dense layer for output probabulity
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#train the model.
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(x_axis_train, 
          y_axis_train, 
          epochs=10, 
          validation_data=(x_axis_test, y_axis_test))

#making predecttion valid job description
model.predict(["responsible for all market operations aspects of mcommerce & driving business metrics of airtel money in secunderabad zone influencing the sales organization ( zbm , zsm , tm , fse ) to deliver all mcommerce kpi"])



