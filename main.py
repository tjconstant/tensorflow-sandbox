import tensorflow as tf
import pandas as pd
from tensorflow import feature_column as fc
from sklearn.model_selection import train_test_split

# Create Dataset
def create_dataset(df, batch_size = 10, shuffle = True):
    dataframe = df.copy()
    dataframe = dataframe.replace(
        {'species' : {
            'Adelie' : 0, 
            'Chinstrap' : 1, 
            'Gentoo' : 2}
            }
            )

    labels = dataframe.pop('species')
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds.shuffle(len(df))
    ds = ds.batch(batch_size)
    return(ds)

# Feature Columns

numerical_cols = [
    'bill_length_mm',
    'bill_depth_mm',
    'flipper_length_mm',
    'body_mass_g'
]

categorical_cols = [
    'island',
    'sex'
]

feature_columns = []
for name in numerical_cols:
    feature_columns.append(fc.numeric_column(name))

HASH_SIZE = 6
for name in categorical_cols:
    feature_columns.append(fc.indicator_column(fc.categorical_column_with_hash_bucket(name, hash_bucket_size = HASH_SIZE)))

from tensorflow.keras import layers

feature_layer = layers.DenseFeatures(feature_columns)

# Load Dataframe
df = pd.read_csv('penguins.csv').dropna()
df.pop('year')

# Dataframe Test/Train
train, test = train_test_split(df)
train, val = train_test_split(train)

# Datasets
ds_train = create_dataset(train)
ds_val = create_dataset(val)
ds_test = create_dataset(test)

# Model (Functional API)

inputs = {
    'bill_length_mm' : layers.Input(shape = (1,), dtype = 'float'),
    'bill_depth_mm' : layers.Input(shape = (1,), dtype = 'float'),
    'flipper_length_mm' : layers.Input(shape = (1,), dtype = 'float'),
    'body_mass_g' : layers.Input(shape = (1,), dtype = 'float'),
    'island' : layers.Input(shape = (1,), dtype = 'string'),
    'sex' : layers.Input(shape = (1,), dtype = 'string')
}

x = feature_layer(inputs)
x = layers.Dense(32, activation = "relu")(x)
x = layers.Dense(64, activation = "relu")(x)
outputs = layers.Dense(3, activation = "sigmoid")(x)

from tensorflow import keras
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

# Fit
history = model.fit(ds_train, epochs = 500, validation_data = ds_val)

model.summary()

model.predict(ds_test)