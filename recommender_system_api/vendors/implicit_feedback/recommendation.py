from recommender_system_api.vendors.implicit_feedback.processing import get_vendor_detail_views_data
from sklearn.model_selection import train_test_split
from recommender_system_api.vendors.implicit_feedback.triplet_neural_network import build_models
from recommender_system_api.vendors.implicit_feedback.evaluation import identity_loss, sample_triplets, average_roc_auc
from config.settings.base import BASE_DIR
import os


def retrain_implicit_vendor_model():
    implicit_df = get_vendor_detail_views_data()
    implicit_df['account_id'] = implicit_df.actual_account_id.astype('category').cat.codes.values
    implicit_df['vendor_id'] = implicit_df.actual_vendor_id.astype('category').cat.codes.values

    train_df, test_df = train_test_split(implicit_df, test_size=0.2, shuffle=True)
    y_train, y_test = train_df.rating.astype('float32').to_numpy(), test_df.rating.astype('float32').to_numpy()

    n_users = implicit_df.account_id.max() + 1
    n_vendors = implicit_df.vendor_id.max() + 1

    hyper_parameters = dict(
        user_dim=32,
        vendor_dim=64, n_hidden=1,
        hidden_size=128,
        dropout=0.1,
        l2_reg=0)
    deep_match_model, deep_triplet_model = build_models(n_users=n_users, n_vendors=n_vendors, **hyper_parameters)
    deep_triplet_model.compile(loss=identity_loss, optimizer='adam')

    n_epochs = 7
    for i in range(n_epochs):
        # Sample new negatives to build different triplets at each epoch
        triplet_inputs = sample_triplets(train_df, n_vendors, random_seed=i)

        # Fit the model incrementally by doing a single pass over the sampled triplets
        deep_triplet_model.fit(triplet_inputs, y_train, shuffle=True, batch_size=32, epochs=1)

        # Monitor the convergence of the model
        test_auc = average_roc_auc(deep_match_model, train_df, test_df)
        print("Epoch %d/%d: test ROC AUC: %0.4f" % (i + 1, n_epochs, test_auc))

    model_path = os.path.join(BASE_DIR, 'recommender_system_api/models/')
    deep_match_model.save(os.path.join(model_path, 'vendor_triplet_model'))

if __name__ == '__main__':
    retrain_implicit_vendor_model()

