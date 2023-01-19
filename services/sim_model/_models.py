import tensorflow as tf
from tensorflow_addons.losses import contrastive_loss
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, first, second):
        distance = tf.reduce_sum(tf.square(first - second), -1)
        return distance

class SiameseModel(tf.keras.Model):
    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        y = data[-1]
        distance = self.siamese_network(data[:-1])
        loss = contrastive_loss(y, distance, margin=self.margin)

        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

def build_emb_model(cnn_layer):
  model = Sequential([
      cnn_layer,
      layers.Flatten(),
      layers.Dense(256, activation='relu'),
      layers.BatchNormalization(),
      layers.Dense(128, activation='relu'),
      layers.BatchNormalization(),
      layers.Dense(128),
  ])

  trainable = False
  for layer in cnn_layer.layers:
      if layer.name == "conv5_block1_out":
          trainable = True
      layer.trainable = trainable

  return model

def build_siamise_network(emb_model, input_shape):
  first_input = layers.Input(name='first', shape=input_shape + (3,))
  second_input = layers.Input(name='second', shape=input_shape + (3,))

  distance = DistanceLayer()(
      emb_model(preprocess_input(first_input)),
      emb_model(preprocess_input(second_input)),
  )

  siamise_network = tf.keras.Model(inputs=[first_input, second_input], outputs=distance)
  return siamise_network


