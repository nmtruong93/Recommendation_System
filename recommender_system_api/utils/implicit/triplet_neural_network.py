from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from recommender_system_api.utils.implicit.evaluation_implicit import margin_comparator_loss, identity_loss
import os
from config.settings.base import BASE_DIR

def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64, dropout=0, l2_reg=None):
    """
    Build the shared multi layer perceptron
    :param input_dim:
    :param n_hidden:
    :param hidden_size:
    :param dropout:
    :param l2_reg:
    :return:
    """
    mlp = Sequential()
    if n_hidden == 0:
        # Plug the output unit directly: this is simple linear regression model. Not dropout required.
        mlp.add(Dense(1, input_dim=input_dim, activation='relu', kernel_regularizer=l2_reg))
    else:
        mlp.add(Dense(hidden_size, input_dim=input_dim, activation='relu', kernel_regularizer=l2_reg))
        mlp.add(Dropout(dropout))
        for i in range(n_hidden-1):
            mlp.add(Dense(hidden_size, activation='relu', W_regularizer=l2_reg))
            mlp.add(Dropout(dropout))
        mlp.add(Dense(1, activation='relu', kernel_regularizer=l2_reg))

    return mlp


def create_model(n_users, n_items, n_genders, n_item_countries, user_dim=32, gender_dim=16, item_dim=64,
                 item_country_dim=16, n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    """
    Build models to train a deep triplet network
    :param n_users:
    :param n_items:
    :param user_dim:
    :param item_dim:
    :param n_hidden:
    :param hidden_size:
    :param dropout:
    :param l2_reg:
    :return:
    """
    user_input = Input((1, ), name='user_input')
    gender_input = Input((1,), name='gender_input')

    positive_item_input = Input((1, ), name='positive_item_input')
    negative_item_input = Input((1, ), name='negative_item_input')

    positive_item_country_input = Input((1, ), name='positive_item_country_input')
    negative_item_country_input = Input((1, ), name='negative_item_country_input')

    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, user_dim, input_length=1, name='user_embedding', embeddings_regularizer=l2_reg)
    gender_layer = Embedding(n_genders, gender_dim, input_length=1, name='gender_embedding', embeddings_regularizer=l2_reg)
    # The following embedding parameters will be shared to encode both the positive and negative items
    item_layer = Embedding(n_items, item_dim, input_length=1, name='item_embedding', embeddings_regularizer=l2_reg)
    item_country_layer = Embedding(n_item_countries, item_country_dim, input_length=1, name='item_country_embedding',
                                   embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    gender_embedding = Flatten()(gender_layer(gender_input))

    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))

    positive_item_country_embedding = Flatten()(item_country_layer(positive_item_country_input))
    negative_item_country_embedding = Flatten()(item_country_layer(negative_item_country_input))

    user_info = Concatenate(name='user_info')([user_embedding, gender_embedding])
    positive_item_info = Concatenate(name='positive_item_info')([positive_item_embedding, positive_item_country_embedding])
    negative_item_info = Concatenate(name='negative_item_info')([negative_item_embedding, negative_item_country_embedding])


    # Similarity computation between embeddings using MLP similarity
    positive_embedding_pair = Concatenate(name='positive_embedding_pair')([user_info, positive_item_info])
    positive_embedding_pair = Dropout(dropout)(positive_embedding_pair)

    negative_embedding_pair = Concatenate(name='negative_embedding_pair')([user_info, negative_item_info])
    negative_embedding_pair = Dropout(dropout)(negative_embedding_pair)

    # Instanciate the shared similarity architecture
    interaction_layers = make_interaction_mlp(user_dim + gender_dim + item_dim + item_country_dim, n_hidden=n_hidden, hidden_size=hidden_size,
                                              dropout=dropout, l2_reg=l2_reg)

    positive_similarity = interaction_layers(positive_embedding_pair)
    negative_similarity = interaction_layers(negative_embedding_pair)

    # The triplet network model, only used for training
    triplet_loss = Lambda(margin_comparator_loss, output_shape=(1, ), name='comparator_loss')\
        ([positive_similarity, negative_similarity])
    deep_triplet_model = Model(inputs=[user_input, gender_input, positive_item_input, negative_item_input,
                                       positive_item_country_input, negative_item_country_input], outputs=[triplet_loss])

    # The match-score model, only used at inference
    deep_match_model = Model(inputs=[user_input, gender_input, positive_item_input, positive_item_country_input],
                             outputs=[positive_similarity])

    deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    return deep_match_model, deep_triplet_model


