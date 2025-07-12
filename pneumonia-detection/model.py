import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import logging

from data import create_data_generators

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 1
LEARNING_RATE = 0.0001
EPOCHS = 25
PATIENCE = 5
MODEL_PATH = 'pneumonia_model.h5'
RESULTS_DIR = 'results'

def build_model():
    """
    Builds the transfer learning model with a ResNet-50 base.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train_model(model, train_generator, validation_generator):
    """
    Trains the model with early stopping and model checkpointing.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    checkpoint = ModelCheckpoint(filepath=MODEL_PATH,
                                 monitor='val_auc',
                                 mode='max',
                                 save_best_only=True,
                                 verbose=1)

    early_stopping = EarlyStopping(monitor='val_auc',
                                   mode='max',
                                   patience=PATIENCE,
                                   verbose=1)

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=validation_generator,
                        callbacks=[checkpoint, early_stopping])
    return history

def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test set and saves the results.
    """
    logger.info("Evaluating model on the test set...")
    model.load_weights(MODEL_PATH) # Load the best model
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(test_generator.class_indices))
    plt.xticks(tick_marks, test_generator.class_indices.keys(), rotation=45)
    plt.yticks(tick_marks, test_generator.class_indices.keys())
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    roc_path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    logger.info(f"ROC curve saved to {roc_path}")

    # Save metrics
    metrics = {
        'test_loss': model.evaluate(test_generator)[0],
        'test_accuracy': model.evaluate(test_generator)[1],
        'test_auc': roc_auc
    }
    with open(os.path.join(RESULTS_DIR, 'test_metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Test metrics saved to {os.path.join(RESULTS_DIR, 'test_metrics.txt')}")


if __name__ == '__main__':
    train_generator, validation_generator, test_generator = create_data_generators()
    model = build_model()
    model.summary()
    history = train_model(model, train_generator, validation_generator)
    evaluate_model(model, test_generator)
    logger.info("Model training and evaluation complete.")
