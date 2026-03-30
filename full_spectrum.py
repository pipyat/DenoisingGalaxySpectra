# Imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
(from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae
from tensorflow.keras.callbacks import History, ReduceLROnPlateau
tf.compat.v1.enable_eager_execution()

# Loading data
flux_train = # define training set here
flux_test =  # define validation set here

# Training parameters
batch_size = 42
epochs = 100
LR_initial = 1e-4
decay_factor = 0.78

# Architectural parameters
latent_dim = 110
act = 'relu'

# Encoder
x = Input(batch_shape = (None,3800,))
dense1 = Dense(3000, activation=act)(x)
dense1 = Dense(2100, activation=act)(dense1)
dense1 = Dense(1500, activation=act)(dense1)
dense1 = Dense(850, activation=act)(dense1)
dense1 = Dense(350, activation=act)(dense1)

z = Dense(latent_dim,activation=act)(dense1)

# Decoder
dense2 = Dense(300, activation=act)(z)
dense2 = Dense(750, activation=act)(dense2)
dense2 = Dense(1400, activation=act)(dense2)
dense2 = Dense(2700, activation=act)(dense2)
dense2 = Dense(3800, activation='linear')(dense2)

full_spectrum = Model(x, dense2)
full_spectrum_encoder = Model(x, z)

# Loss function
def loss(y_true, y_pred):
    xent_loss = mae(y_true, y_pred)
    return xent_loss

# Define callbacks
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=decay_factor,
    patience=4,
    min_lr=1e-7,
    verbose=1)
history = History()

# Compile the model with the optimiser and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=LR_initial) 
full_spectrum.compile(optimizer=optimizer, loss=loss)

print(full_spectrum.summary())

# Fit the model with callbacks
full_spectrum.fit(
    flux_train,
    flux_train,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(flux_test, flux_test),

    callbacks=[history, reduce_lr_callback]
)
