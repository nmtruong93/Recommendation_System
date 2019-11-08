from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
from config.settings.base import BASE_DIR

# TODO: Integrate metadata into model, GridSearch to optimize model
def create_model(params):
    user_input = Input(shape=[1], name='User-Input')
    vendor_input = Input(shape=[1], name='Vendor-Input')
    gender_input = Input(shape=[1], name='Gender-Input')
    vendor_country_input = Input(shape=[1], name='Vendor-Country-Input')

    # Create vendor embedding path
    user_embedding = Embedding(input_dim=params['n_users'] + 1,
                               output_dim=params['n_latent_factors'], name='User-Embedding')(user_input)
    vendor_embedding = Embedding(input_dim=params['n_vendors'] + 1,
                                 output_dim=params['n_latent_factors'], name='Vendor-Embedding')(vendor_input)
    gender_embedding = Embedding(input_dim=params['n_genders'] + 1,
                                 output_dim=params['n_latent_factors'], name='Gender-Embedding')(gender_input)
    vendor_country_embedding = Embedding(input_dim=params['n_vendor_countries'] + 1,
                                         output_dim=params['n_latent_factors'], name='Vendor-Country-Embedding')(vendor_country_input)
    # Flatten embedding
    user_vec = Flatten(name='Flatten-Users')(user_embedding)
    vendor_vec = Flatten(name='Flatten-Vendors')(vendor_embedding)
    gender_vec = Flatten(name='Flatten-Genders')(gender_embedding)
    vendor_country_vec = Flatten(name='Flatten-Vendor-Countries')(vendor_country_embedding)

    user_vec = Dense(64)(user_vec)
    vendor_vec = Dense(64)(vendor_vec)
    gender_vec = Dense(16)(gender_vec)
    vendor_country_vec = Dense(16)(vendor_country_vec)

    # Concatenate feature
    user_info = Concatenate()([user_vec, gender_vec])
    vendor_info = Concatenate()([vendor_vec, vendor_country_vec])
    input_vectors = Concatenate()([user_info, vendor_info])

    # Add fully-connected layers
    fc1 = Dense(128, activation='relu')(input_vectors)
    batch_1 = BatchNormalization()(fc1)
    dropout_1 = Dropout(rate=0.2)(batch_1)

    out = Dense(1)(dropout_1)
    model = Model([user_input, gender_input, vendor_input, vendor_country_input], out)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model


def train_model(x_train, y_train, model_path, model):
    checkpoint_path = os.path.join(model_path, 'vendor_neural_net.h5')
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1,
                                       save_best_only=True, mode='auto', save_weights_only=True)
    tensor_board = TensorBoard(log_dir=os.path.join(BASE_DIR, 'logs'), histogram_freq=0, write_graph=True, write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,
                                   mode='auto', restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=100, verbose=1,
                        callbacks=[model_checkpoint, early_stopping, tensor_board])
    return model, history

