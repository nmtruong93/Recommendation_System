import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class NeuralNetwork(object):

    def __init__(self, rating_df, train_df, n_latent_factors):
        self.rating_df = train_df
        self.n_latent_factors = n_latent_factors
        self.n_users = rating_df.user_id.max()
        self.n_vendors = rating_df.vendor_id.max()
        self.n_genders = rating_df.gender.max()
        self.n_vendor_countries = rating_df.vd_country_id.max()

    # TODO: Integrate metadata into model, GridSearch to optimize model
    def model(self, file_path):
        user_input = Input(shape=[1], name='User-Input')
        vendor_input = Input(shape=[1], name='Vendor-Input')
        gender_input = Input(shape=[1], name='Gender-Input')
        vendor_country_input = Input(shape=[1], name='Vendor-Country-Input')

        # Create vendor embedding path
        user_embedding = Embedding(input_dim=self.n_users + 1,
                                   output_dim=self.n_latent_factors, name='User-Embedding')(user_input)
        vendor_embedding = Embedding(input_dim=self.n_vendors + 1,
                                     output_dim=self.n_latent_factors, name='Vendor-Embedding')(vendor_input)
        gender_embedding = Embedding(input_dim=self.n_genders + 1,
                                     output_dim=self.n_latent_factors, name='Gender-Embedding')(gender_input)
        vendor_country_embedding = Embedding(input_dim=self.n_vendor_countries + 1,
                                             output_dim=self.n_latent_factors, name='Vendor-Country-Embedding')(vendor_country_input)
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

        # fc2 = Dense(64, activation='relu')(dropout_1)
        # batch_2 = BatchNormalization()(fc2)
        # dropout_2 = Dropout(rate=0.2)(batch_2)

        out = Dense(1)(dropout_1)
        model = Model([user_input, gender_input, vendor_input, vendor_country_input], out)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        checkpoint_path = file_path + '/vendor_neural_net.h5'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='auto', save_weights_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1,
                                       mode='auto', restore_best_weights=True)

        model.fit(
            x=[self.rating_df.user_id, self.rating_df.gender, self.rating_df.vendor_id, self.rating_df.vd_country_id],
            y=self.rating_df.rating,
            validation_split=0.2, epochs=100, verbose=1, callbacks=[model_checkpoint, early_stopping])
        return model

    @staticmethod
    def plot_loss_accuracy(history, file_path='../../models/training_loss.png'):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(file_path)
