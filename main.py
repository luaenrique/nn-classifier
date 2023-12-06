import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from river import base, drift
import random

class GenericNewClassifier(base.Classifier):
    def __init__(self, n_classifiers: int, hidden_units_range=(2, 10), learning_rate_range=(0.001, 0.01), max_instances_after_drift=1000, min_accuracy_change=0.05, max_classifiers_creation=2, max_classifiers=15):
        self.n_classifiers = n_classifiers
        self.hidden_units_range = hidden_units_range
        self.learning_rate_range = learning_rate_range
        self.input_shape = None
        self.classifiers = []
        self.samples_seen = 0
        self.correct_predictions = 0
        self.adwin_detector = drift.ADWIN()
        self.instances_after_drift = 0
        self.max_instances_after_drift = max_instances_after_drift
        self.min_accuracy_change = min_accuracy_change
        self.accuracy_window = []
        self.window_count = 0
        self.initial_loss = 0
        self.window_to_create_new_count = 0
        self.worst_classifier_index = None
        self.losses = []
        self.max_classifiers_creation = max_classifiers_creation
        self.max_classifiers = max_classifiers


    import random

    def _create_mlp_model(self, input_shape, transfer_from=None):
        # Randomly choose parameters for each MLP
        denominator = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
        numerator = [5.0]

        # Generate learning rate as a division of numerator and denominator
        learning_rate_numerator = np.random.choice(numerator)
        learning_rate_denominator = np.random.choice(denominator)
        learning_rate = learning_rate_numerator / learning_rate_denominator

        model = tf.keras.Sequential()

        # If transfer learning is specified, add and freeze layers from the existing model
        if transfer_from is not None:
            for i, layer in enumerate(transfer_from.layers):
                model.add(layer)
                if i < len(transfer_from.layers) - 1:  # Freeze all layers except the last one
                    layer.trainable = False

        # Add hidden layers
        num_hidden_layers = random.randint(1, 10)
        for _ in range(num_hidden_layers):
            units_per_layer = random.randint(2, 64)
            dropout_rate = random.uniform(0.4, 0.9)

            model.add(Dense(units=units_per_layer, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=1, activation='sigmoid'))

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

        # Get predictions and losses from all classifiers
        predictions = []
        losses = []
        for classifier in self.classifiers:
            prediction = classifier.predict(x_formatted)
            loss = classifier.evaluate(x_formatted, self._convert_label_to_input_format(0))
            predictions.append(prediction)
            losses.append(loss)
            

        # Find the index of the classifier with the minimum loss
        min_loss_index = np.argmin(losses)
        max_loss_index = np.argmax(losses)

        # Use the classifier with the minimum loss for the final prediction
        final_prediction = 1 if predictions[min_loss_index][0] >= 0.5 else 0

        # Update correct predictions count
        self.correct_predictions += final_prediction == self._convert_label_to_input_format(1)

        # Update ADWIN detector
        self.adwin_detector.update(final_prediction)
        
        self.initial_loss = losses[min_loss_index]
        
        if self.window_to_create_new_count == 0:
            self.worst_classifier_index = max_loss_index
            self.initial_loss = losses[self.worst_classifier_index]
        if self.window_to_create_new_count == 100:
            self.window_to_create_new_count = 0
            if self.initial_loss < losses[self.worst_classifier_index]:
                transfer_knowledge_from = self.classifiers[min_loss_index]
                self.classifiers.pop(self.worst_classifier_index)
                losses.pop(self.worst_classifier_index)
                
                new_classifiers = 0
                
                while (len(self.classifiers) != self.max_classifiers and new_classifiers < self.max_classifiers_creation):
                    new_classifier = self._create_mlp_model(self.input_shape, transfer_from=transfer_knowledge_from)
                    self.classifiers.append(new_classifier)
                    new_classifiers += 1
                    print('criei um novo mlp')
        
        self.window_to_create_new_count += 1
        
        # Check for concept drift
        if self.adwin_detector.change_detected:
            print("Concept drift detected!")
            self.instances_after_drift = 0  # Reset the count
            self.accuracy_window = []  # Reset the accuracy window
            
        


        else:
            self.instances_after_drift += 1
            self.accuracy_window.append(final_prediction)

            # Check if the drift has persisted for too long
            if self.instances_after_drift >= self.max_instances_after_drift:
                print(f"Drift has persisted for {self.instances_after_drift} instances.")

                # Check if accuracy changed significantly
                current_accuracy = self.correct_predictions / self.samples_seen
                avg_accuracy_in_window = np.mean(self.accuracy_window)

                if abs(current_accuracy - avg_accuracy_in_window) <= self.min_accuracy_change:
                    print(f"Accuracy didn't change significantly within the window. Taking action...")

                    # Add your custom actions or model adjustments here

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


from river import stream
dataset = []
for x, y in stream.iter_arff('electricity.arff', target='class'):
    if y == 'UP':
        dataset.append((x, 1))
    else:
        dataset.append((x, 0))


from river import evaluate
from river import metrics

model = GenericNewClassifier(n_classifiers=10)

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric, print_every=1000)