"""
initial source code: Khalid Salama
```python
pip install -U tensorflow-addons
```
"""

"""
## Setup
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import os
from email import generator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa
from data_preprocessing import My_Custom_Generator
# gpus = tf.config.experimental.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



"""
## Prepare the data
"""
## Configure the hyperparameters
num_classes = 2
checkpoint_dir = "Pre_training/checkpoint/cp.ckpt"
input_shape = (224, 224, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 5
image_size = 224  # We'll resize input images to this size
patch_size = 32  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


"""
## Implement multilayer perceptron (MLP)
"""


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


"""
## Implement patch creation as a layer
"""


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""
Let's display patches for a sample image
"""
def dispaly_sample(x_train):
    plt.figure(figsize=(4, 4))
    random_index = np.random.choice(range(len(x_train)))
    print('random_index: ',random_index)
    image = x_train[random_index]
    plt.imshow(image.astype("uint8"))
    plt.axis("off")

    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]), size=(image_size, image_size)
    )

    patches = Patches(patch_size)(resized_image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
        plt.show()

"""
## Implement the patch encoding layer

The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded




def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


"""
## Compile, train, and evaluate the mode
"""


def run_experiment(model,training_batch_generator,validation_batch_generator,test_batch_generator,num_epochs):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_path = "Pre_training/checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    #     # Save the weights using the `checkpoint_path` format
  # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    # history = model.fit(
    #     generator=training_batch_generator,
    #     steps_per_epoch= 100,#(len(training_batch_generator)), #),#100,
    #     verbose=1,
    #     epochs=num_epochs,
    #     validation_data=validation_batch_generator,
    #     validation_steps=50,#(len(validation_batch_generator) // batch_size),
    #     callbacks=[checkpoint_callback],
    # )


    history = model.fit(
        x=training_batch_generator,
        steps_per_epoch=(len(training_batch_generator)), #),#100,
        batch_size=batch_size,
        verbose=1,
        epochs=num_epochs,
        validation_data=validation_batch_generator,
        validation_steps=(len(validation_batch_generator) // batch_size),
        callbacks=[checkpoint_callback],
    )
    #keras.models.save_model(model, 'saved_model/my_model.h5')
    # keras.model.save('saved_model/my_model.h5')
    # model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(test_batch_generator, steps=2000, verbose=2)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    probabilities = model.predict(test_batch_generator)
    y_true = test_batch_generator.labels
    #print("y_true.shape",y_true.shape)
    #print("probabilities.shape",probabilities.shape)
    y_pred= np.argmax(probabilities,axis=1)
    #print("probabilities.shape after",y_pred.shape)
    
    mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    # plot_confusion_matrix(mat, X, y)
    plt.show()
    score = f1_score(y_true, y_pred, average=None)
    # Print f1, precision, and recall scores
    precision = precision_score(y_true, y_pred , average="binary")
    recall = recall_score(y_true, y_pred , average="binary")
    score = f1_score(y_true, y_pred , average="binary")
    print(f"Test precision: {precision}%")
    print(f"Test recall: {recall}%")
    print(f"Test score: {score}%")



    return history,model


def main():
    # loading filenames and labels for training
    loaded_fname = np.load('PIE_dataset/train_filenames.npy')
    loaded_labels = np.load('PIE_dataset/train_labels.npy')

    training_batch_generator = My_Custom_Generator(loaded_fname,loaded_labels,batch_size)    
    sample = training_batch_generator.__getitem__(0)[0][0] 
    # lst_str = str(lst)[1:-1]    
    print('Training number of batches: ' ,len(training_batch_generator), ' Batch size : ',batch_size, 'Image shape: ',sample.shape)#, [length for length in sample.shape()])
    print('Training Data Shape: ' ,(len(training_batch_generator),batch_size,sample.shape))
    
    print('-----------------------------------------------------------------------------------------')
    loaded_fname_val = np.load('PIE_dataset/val_filenames.npy')
    loaded_labels_val = np.load('PIE_dataset/val_labels.npy')
    validation_batch_generator = My_Custom_Generator(loaded_fname_val,loaded_labels_val,batch_size) 
    sample_val = validation_batch_generator.__getitem__(0)[0][0]
    print('validation number of batches: ' ,len(validation_batch_generator), ' Batch size : ',batch_size, 'Image shape: ',sample_val.shape)#, [length for length in sample.shape()])
    print('validation Data Shape: ' ,(len(validation_batch_generator),batch_size,sample_val.shape))

    print('-----------------------------------------------------------------------------------------')
    loaded_fname_test = np.load('PIE_dataset/test_filenames.npy')
    loaded_labels_test = np.load('PIE_dataset/test_labels.npy')
 
    test_batch_generator = My_Custom_Generator(loaded_fname_test,loaded_labels_test,batch_size) 
    sample_test = test_batch_generator.__getitem__(0)[0][0]
    print('Test number of batches: ' ,len(test_batch_generator), ' Batch size : ',batch_size, 'Image shape: ',sample_test.shape)#, [length for length in sample.shape()])
    print('Test Data Shape: ' ,(len(test_batch_generator),batch_size,sample_test.shape))


    return training_batch_generator,validation_batch_generator,test_batch_generator

if __name__ == '__main__':
    pretrained=False
    fix_gpu()
    checkpoint_path="Pre_training/latest_one"

    training_gen, validation_gen,test_batch_gen= main()
    #dispaly_sample(training_gen.__getitem__(0)[0])
    vit_classifier = create_vit_classifier()
    if pretrained==True:
        # Loads the weights
        # vit_classifier.load_weights(checkpoint_path)
        model= tf.keras.models.load_model('saved_model/my_model.h5')
            # Re-evaluate the model
    # model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = vit_classifier.evaluate(test_batch_gen, steps=2000, verbose=2)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    else:
        history, model = run_experiment(vit_classifier,training_gen,validation_gen,test_batch_gen,num_epochs)
        # loss, acc = vit_classifier.evaluate_generator(test_batch_generator, steps=(len(test_batch_generator) // batch_size), verbose=2)
        # print("Trained model, accuracy: {:5.2f}%".format(100 * acc))
    
