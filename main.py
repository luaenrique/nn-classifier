import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from river import base

class GenericClassifier(base.Classifier):
    def __init__(self, n_classifiers: int, hidden_units: int):
        self.n_classifiers = n_classifiers
        self.hidden_units = hidden_units
        self.input_shape = None
        self.classifiers = []

    def _create_mlp_model(self, input_shape):
        model = tf.keras.Sequential()
        model.add(Dense(units=self.hidden_units, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def _initialize_classifiers(self, x):
        if self.input_shape is None:
            features = len(x)
            self.input_shape = (features,)

        self.classifiers = [self._create_mlp_model(self.input_shape) for _ in range(self.n_classifiers)]

    def _convert_to_input_format(self, x):
        return np.array([list(x.values())])

    def _convert_label_to_input_format(self, y):
        return np.array([y])

    def predict_proba_one(self, x, **kwargs):
        if not self.classifiers:
            self._initialize_classifiers(x)

        x_formatted = self._convert_to_input_format(x)
        return self.classifiers[0].predict(x_formatted)

    def predict_one(self, x, **kwargs):
        if not self.classifiers:
            self._initialize_classifiers(x)

        x_formatted = self._convert_to_input_format(x)
        y_pred = self.classifiers[0].predict(x_formatted)
        return 1 if y_pred[0] >= 0.5 else 0

    def _train_single_classifier(self, classifier, x_train, y_train, epochs):
        for epoch in range(epochs):
            classifier.fit(x_train, y_train, epochs=1, verbose=0)
            # Simulate some work being done during training if needed

    def learn_one(self, x, y, epochs=1, **kwargs):
        if not self.classifiers:
            self._initialize_classifiers(x)

        x_formatted = self._convert_to_input_format(x)
        y_formatted = self._convert_label_to_input_format(y)

        # Assuming features is the length of x, you might need to adjust this accordingly
        x_formatted = x_formatted.reshape((1, len(x)))

        # Create and start threads for each classifier's training
        threads = []
        for classifier in self.classifiers:
            thread = threading.Thread(target=self._train_single_classifier, args=(classifier, x_formatted, y_formatted, epochs))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        return self
