import pandas as pd
from tensorflow.keras.models import load_model
from recommender_system_api.utils.implicit.data_getting_processing import get_full_data
from recommender_system_api.utils.implicit.triplet_neural_network import create_model
from recommender_system_api.utils.implicit.evaluation_implicit import sample_triplets
from config.settings.base import BASE_DIR
import os


def retrain_implicit_model(vendor=True):

    implicit_df = get_full_data(vendor=vendor)
    implicit_df['account_id'] = implicit_df.actual_account_id.astype('category').cat.codes.values
    implicit_df['item_id'] = implicit_df.actual_item_id.astype('category').cat.codes.values
    implicit_df['item_country_id'] = implicit_df.actual_item_country_id.astype('category').cat.codes.values

    n_users = implicit_df.account_id.max() + 1
    n_items = implicit_df.item_id.max() + 1
    n_genders = implicit_df.gender.max() + 1
    n_item_countries = implicit_df.item_country_id.max() + 1

    y_train = implicit_df.rating.astype('float32').to_numpy()

    # Should tune these hyper-parameters
    hyper_parameters = dict(
        user_dim=32,
        gender_dim=16,
        item_dim=64,
        item_country_dim=16,
        n_hidden=1,
        hidden_size=128,
        dropout=0.1,
        l2_reg=0)

    deep_match_model, deep_triplet_model = create_model(n_users=n_users, n_items=n_items, n_genders=n_genders,
                                                        n_item_countries=n_item_countries, **hyper_parameters)
    # deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    n_epochs = 15
    val_loss_list = []
    match_model_list = []
    for i in range(n_epochs):
        # Sample new negatives to build different triplets at each epoch
        triplet_inputs = sample_triplets(implicit_df, n_items, random_seed=i)

        # Fit the model incrementally by doing a single pass over the sampled triplets
        history = deep_triplet_model.fit(triplet_inputs, y_train, validation_split=0.2, shuffle=True, batch_size=32,
                                         epochs=4)
        # Early stopping with patience = 3
        break_loop = False
        for j in range(len(val_loss_list), 0, -1):
            if history.history['val_loss'][-1] >= val_loss_list[j-1]:
                if j-1 <= len(val_loss_list) - 3:
                    last_three_list = history.history['val_loss'][-1] > val_loss_list[-3: ]
                    if False not in last_three_list and sorted(val_loss_list[-3: ]) == val_loss_list[-3: ]:
                        break_loop = True
                        print("STOPP ======================================")
                        break
        if break_loop:
            break

        val_loss_list.append(history.history['val_loss'][0])
        match_model_list.append(deep_match_model)

    # Take the model with highest auc score to make prediction
    highest_auc_index = val_loss_list.index(max(val_loss_list))
    deep_match_model = match_model_list[highest_auc_index]

    model_path = os.path.join(BASE_DIR, 'recommender_system_api/models/')
    if vendor:
        deep_match_model.save(os.path.join(model_path, 'vendor_triplet_model.h5'))
        implicit_df.to_feather(os.path.join(model_path, 'implicit_vendor_df.feather'))
    else:
        deep_match_model.save(os.path.join(model_path, 'coupon_triplet_model.h5'))
        implicit_df.to_feather(os.path.join(model_path, 'implicit_coupon_df.feather'))

    print("="*200)
    print("DONE RETRAIN IMPLICIT MODEL")
    print("=" * 200)


def load_models(vendor=True):
    """
    Get the results from all previous processing
    :return:
    """
    model_path = os.path.join(BASE_DIR, 'recommender_system_api/models/')
    if vendor:
        triplet_model = load_model(os.path.join(model_path, 'vendor_triplet_model.h5'))
        implicit_df = pd.read_feather(os.path.join(model_path, 'implicit_vendor_df.feather'))
    else:
        triplet_model = load_model(os.path.join(model_path, 'coupon_triplet_model.h5'))
        implicit_df = pd.read_feather(os.path.join(model_path, 'implicit_coupon_df.feather'))

    return implicit_df, triplet_model