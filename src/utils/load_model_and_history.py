import keras
import os
import pickle


def load_model_and_history(name,
                           data_type='mnist'):
    folder_path = os.path.join("../../models/" + data_type, name)

    # Load the model
    model_file_path = os.path.join(folder_path, 'model.keras')
    model = keras.saving.load_model(model_file_path)

    # Load history from history.pkl file
    path = os.path.join(folder_path, 'history.pkl')
    with open(path, "rb") as file_pi:
        history = pickle.load(file_pi)

    return model, history