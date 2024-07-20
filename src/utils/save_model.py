import os


def save_model(model,
               folder):
    
    folder_path = os.path.join("../../models/mnist", folder)
    os.makedirs(folder_path, exist_ok=True)
    model_file_path = os.path.join(folder_path, 'model.keras') # .keras is recommended keras 3 format (see https://www.tensorflow.org/guide/keras/serialization_and_saving#registering_the_custom_object)
    model.save(model_file_path)