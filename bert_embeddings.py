import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.models import Model


tf.random.set_seed(16)


id = "1t_bpJX7PC0dqZ5pEmaxM8wmDJteYmOXB" # google file ID
df = pd.read_csv("https://docs.google.com/uc?id=" + id, sep = ';')

df = df[['topic', 'title']]
df.head()

labels_dict = {
    'SCIENCE': 0, 
    'TECHNOLOGY': 1, 
    'HEALTH': 2, 
    'WORLD': 3, 
    'ENTERTAINMENT': 4,
    'SPORTS': 5, 
    'BUSINESS': 6, 
    'NATION': 7,
}
df.topic = df.topic.apply(lambda x: labels_dict[x])

train_val_df, test_df = train_test_split(df, test_size=0.5, random_state=12)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=12)

train_ds = tf.data.Dataset.from_tensor_slices((train_df.title, train_df.topic))
val_ds = tf.data.Dataset.from_tensor_slices((val_df.title, val_df.topic))
test_ds = tf.data.Dataset.from_tensor_slices((test_df.title, test_df.topic))

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=True)

inputs = keras.Input(shape=(), dtype=tf.string, name='sentences')
preprocessed_text = bert_preprocess(inputs)
preprocessed_text = bert_encoder(preprocessed_text)
x = keras.layers.Dropout(0.1, name="dropout")(preprocessed_text['pooled_output'])
outputs = tf.keras.layers.Dense(8, activation='softmax', name="output")(x)
model = keras.Model(inputs, outputs)

optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0
)

model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("bert.keras",
                                             save_best_only=True)]
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

model = keras.models.load_model("bert.keras",
                                custom_objects={"KerasLayer": hub.KerasLayer})

XX = model.input 
YY = model.layers[2].output
new_model = Model(XX, YY)

embeddings = []
topics = []
for x in test_ds:
    out = new_model.predict(x[0])
    embeddings.extend(out['pooled_output'].tolist())
    topics.extend(x[1].numpy().tolist())

df_emb = pd.DataFrame(embeddings)
df_emb['topic'] = pd.Series(topics)
df_emb.to_csv('news_embeddings.csv', index=False)
