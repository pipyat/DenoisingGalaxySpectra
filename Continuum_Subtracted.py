# Imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae
from tensorflow.keras.callbacks import History, ReduceLROnPlateau
tf.compat.v1.enable_eager_execution()

# Loading data
flux_train_sub = # define continuum subtracted training set here 
flux_train_full = # define ordinary spectra training set here 
flux_test_sub =  # define continuum subtracted validation set here 
flux_test_full =  # define continuum subtracted validation set here 

# Training parameters
batch_size = 42
epochs = 100
LR_initial = 1e-4
decay_factor = 0.78

# Architectural parameters
latent_dim = 400

# Encoder
x = Input(batch_shape = (None,3800,))
dense1 = Dense(2800)(x)
dense1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense1)
dense1 = Dense(1200)(dense1)
dense1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense1)
dense1 = Dense(800)(dense1)
dense1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense1)
dense1 = Dense(450)(dense1)
dense1 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense1)

z = Dense(latent_dim)(dense1)
z = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(z)

# Decoder
dense2 = Dense(450)(z)
dense2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense2)
dense2 = Dense(800)(dense2)
dense2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense2)
dense2 = Dense(1100(dense2)
dense2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense2)
dense2 = Dense(1600)(dense2)
dense2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense2)
dense2 = Dense(2800)(dense2)
dense2 = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.2))(dense2)
dense2 = Dense(3800)(dense2)

continuum_subtracted = Model(x, dense2)
continuum_subtracted_encoder = Model(x, z)

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
continuum_subtracted.compile(optimizer=optimizer, loss=loss)

print(continuum_subtracted.summary())

# Fit the model with callbacks
continuum_subtracted.fit(
    flux_train_sub,
    flux_train_full,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(flux_test_sub, flux_test_full),

    callbacks=[history, reduce_lr_callback]
)
