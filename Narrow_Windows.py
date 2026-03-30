# Imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate 
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mae
from tensorflow.keras.callbacks import History, ReduceLROnPlateau
tf.compat.v1.enable_eager_execution()

# Loading data
r_train = # define red region training set here
r_test = # define red region validation set here
b_train = # define blue region training set here
b_test = # define blue region validation set here
flux_train = # define ordinary spectra training set here
flux_test = # define ordinary spectra validation set here

# Training parameters
batch_size = 42
epochs = 100
LR_initial = 1e-4
decay_factor = 0.78

# Architectural parameters
act = 'relu'

# Forked model, layers named for clarity in summary
input_b = Input(shape=(400,), name="blue_region_input")
input_r = Input(shape=(400,), name="red_region_input")

# First branch for the blue region (b_region)
x_b1 = Dense(400, activation=act, name="dense_b_1")(input_b)
x_b = Dense(800, activation=act, name="dense_b_2")(x_b1)
x_b = Dense(1100, activation=act, name="dense_b_3")(x_b)

# Second branch for the red region (r_region)
x_r1 = Dense(400, activation=act, name="dense_r_1")(input_r)
x_r = Dense(800, activation=act, name="dense_r_2")(x_r1)
x_r = Dense(1100, activation=act, name="dense_r_3")(x_r)

# Concatenate the outputs of the two branches
merged = Concatenate(name="concat")([x_b, x_r])

# Fully connected layers after concatenation
x = Dense(1800, activation=act, name="dense_merged_1")(merged)
x = Dense(2100, activation=act, name="dense_merged_2")(x)
x = Dense(2400, activation=act, name="dense_merged_3")(x)
x = Dense(2800, activation=act, name="dense_merged_4")(x)
x = Dense(3000, activation=act, name="dense_merged_5")(x)
x = Dense(3400, activation=act, name="dense_merged_6")(x)
output = Dense(3800, activation="linear", name="output_layer")(x)

narrow_windows = Model(inputs=[input_b, input_r], outputs=output, name="TwoHeadedSpectraModel")

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
narrow_windows.compile(optimizer=optimizer, loss=loss)

print(narrow_windows.summary())

# Fit the model with callbacks
history = narrow_windows.fit(
    x=[b_train, r_train],
    y=flux_train,
    validation_data=([b_test, r_test], flux_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[history,reduce_lr_callback]
)
