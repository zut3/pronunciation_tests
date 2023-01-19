import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from ._models import build_emb_model, build_siamise_network, SiameseModel
from ._utils import load_audio
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(1,5))

target_shape = (321, 129)
emb_model = build_emb_model(ResNet50(weights='imagenet', include_top=False, input_shape=target_shape + (3,)))
siamise_net = build_siamise_network(emb_model, target_shape)

def _get_model():
    model = SiameseModel(siamise_net)
    model.built = True
    model.load_weights("services/sim_model/weights.h5")
    return model

def predict(first_paths, second_paths):
    assert len(first_paths) == len(second_paths)
  
    model = _get_model()
    predicted = []
    for first_path, second_path in zip(first_paths, second_paths):
        first = tf.expand_dims(load_audio(first_path, target_shape), 0)
        second = tf.expand_dims(load_audio(second_path, target_shape), 0)
    
        dist = model.predict([first, second])[0]
        predicted.append(dist)
    
    res = scaler.fit_transform((1 - np.array(predicted)).reshape(-1,1))
    return np.concatenate(res, 0).astype(np.int32)

