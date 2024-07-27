import os
import pickle


def save_model_and_history(model,
                           history,  
                           save_name,
                           data_type='mnist'):  # 'mnist' or 'vol'
    
    folder_path = os.path.join("../../models/" + data_type, save_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the model
    model_file_path = os.path.join(folder_path, 'model.keras') # .keras is recommended keras 3 format (see https://www.tensorflow.org/guide/keras/serialization_and_saving#registering_the_custom_object)
    model.save(model_file_path)

    # Save history to the history.pkl file
    hist_file_path = os.path.join(folder_path, 'history.pkl')
    with open(hist_file_path, 'wb') as file_pi:
        pickle.dump(history, file_pi)


