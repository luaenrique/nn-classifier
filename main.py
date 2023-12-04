import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from river import base

class GenericClassifier(base.Classifier):
    def __init__(self, n_classifiers: int, hidden_units_range=(2, 5, 10, 100), learning_rate_range=(0.00001, 0.001, 0.01)):
        self.n_classifiers = n_classifiers
        self.hidden_units_range = hidden_units_range
        self.learning_rate_range = learning_rate_range
        self.input_shape = None
        self.classifiers = []
        self.samples_seen = 0
        self.correct_predictions = 0

    def _create_mlp_model(self, input_shape):
        # Randomly choose parameters for each MLP
        hidden_units = np.random.randint(self.hidden_units_range[0], self.hidden_units_range[1])
        learning_rate = np.random.uniform(self.learning_rate_range[0], self.learning_rate_range[1])

        model = tf.keras.Sequential()
        model.add(Dense(units=hidden_units, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Use the chosen learning rate for optimization
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
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

        # Get predictions from all classifiers
        predictions = [classifier.predict(x_formatted) for classifier in self.classifiers]

        # Find the index of the classifier with the minimum loss
        min_loss_index = np.argmin([classifier.evaluate(x_formatted, self._convert_label_to_input_format(0)) for classifier in self.classifiers])

        # Use the classifier with the minimum loss for final prediction
        final_prediction = 1 if predictions[min_loss_index][0] >= 0.5 else 0

        # Update correct predictions count
        self.correct_predictions += final_prediction == self._convert_label_to_input_format(1)

        return final_prediction

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
            
        self.samples_seen += 1

        accuracy = self.correct_predictions / self.samples_seen

        return self



from river.datasets import synth

dataset = synth.ConceptDriftStream(
    stream=synth.LED(seed=42),
    drift_stream=synth.LED(seed=42),
    seed=1, position=125000, width=5000
).take(10000)



from river import evaluate
from river import metrics

model = GenericClassifier(n_classifiers=10)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric, print_every=10000)