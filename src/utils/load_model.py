import keras
import os


def load_model(folder):
    model_file_path = os.path.join("../../models/mnist", folder, 'model.keras')
    return keras.saving.load_model(model_file_path)