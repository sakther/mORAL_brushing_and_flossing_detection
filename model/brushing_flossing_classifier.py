from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd

selected_features_brushing = ['ay:zero_crossing', 'az:fft_centroid', 'max_accl:spec_rolloff', 'ax:fft_centroid',
                              'max_accl:zero_crossing', 'max_accl:fft_centroid', 'az:zero_crossing',
                              'max_accl:spec_entropy', 'ax:spec_entropy', 'ax:spec_rolloff', 'gx:zero_crossing',
                              'ax:zero_crossing', 'ay:mean', 'ay:median', 'ay:power', 'ay:fft_centroid',
                              'gz:zero_crossing', 'gz:fft_centroid', 'gz:spec_rolloff', 'duration', 'ax:std', 'ax:kurt',
                              'gx:kurt', 'gx:fft_centroid', 'gx:spec_rolloff']
selected_feature_index_brushing = [20, 34, 52, 8, 46, 47, 33, 50, 11, 13, 59, 7, 14, 15, 19, 21, 85, 86, 91, 0, 3, 5,
                                   57, 60, 65]
brushing_model_file_name = 'model\AB_model_brushing_selected_features.model'

selected_features_flossing = ['ay:mean', 'ay:median', 'ay:power', 'yaw:median', 'roll:power', 'yaw:mean', 'ax:power',
                              'ax:spec_entropy', 'ay:spec_entropy', 'az:power', 'gy:std', 'gy:skewness', 'gy:power',
                              'roll:median', 'roll:std', 'yaw:power', 'c_g_mag', 'mse_roll', 'mse_yaw', 'duration',
                              'ax:mean', 'ax:median', 'ax:std', 'ax:fft_centroid', 'ay:std']
selected_feature_index_flossing = [14, 15, 19, 119, 97, 118, 6, 11, 24, 32, 68, 69, 71, 93, 94, 123, 147, 148, 150, 0,
                                   1, 2, 3, 8, 16]
flossing_model_file_name = 'model\AB_model_flossing_selected_features.model'


def get_model(model_file_name) -> AdaBoostClassifier:
    """
    :rtype: object
    :return:
    """
    with open(model_file_name, 'rb') as handle:
        clf = pickle.load(handle)
    # clf = pickle.loads(model_file_name)
    return clf


def classify_brushing(X: pd.DataFrame):
    X = X[selected_features_brushing]
    clf = get_model(brushing_model_file_name)
    preds = clf.predict(X.values)

    return preds


def classify_flossing(X: pd.DataFrame):
    X = X[selected_features_flossing]
    clf = get_model(flossing_model_file_name)
    preds = clf.predict(X.values)

    return preds
