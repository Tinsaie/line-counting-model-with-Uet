BATCH_SIZE = 4
EPOCHS = 50

from sklearn.model_selection import train_test_split
from tensorflow.keras import losses, metrics, callbacks

# Split
train_img, val_img, train_msk, val_msk = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

# Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def bce_dice_loss(y_true, y_pred):
    return losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

# Generator
train_gen = data_generator(train_img, train_msk, augment=True)
val_gen = data_generator(val_img, val_msk, augment=False)

# Compile
model.compile(optimizer='adam',
              loss=bce_dice_loss,
              metrics=[dice_loss, metrics.MeanIoU(num_classes=2)])

# Callbacks
checkpoint = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=50,
                    steps_per_epoch=len(train_img)//BATCH_SIZE,
                    validation_steps=len(val_img)//BATCH_SIZE,
                    callbacks=[checkpoint, early_stop])


BATCH_SIZE = 4
EPOCHS = 50

from sklearn.model_selection import train_test_split
from tensorflow.keras import losses, metrics, callbacks

# Split
train_img, val_img, train_msk, val_msk = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

# Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def bce_dice_loss(y_true, y_pred):
    return losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

# Generator
train_gen = data_generator(train_img, train_msk, augment=True)
val_gen = data_generator(val_img, val_msk, augment=False)

# Compile
model.compile(optimizer='adam',
              loss=bce_dice_loss,
              metrics=[dice_loss, metrics.MeanIoU(num_classes=2)])

# Callbacks
checkpoint = callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
early_stop = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=50,
                    steps_per_epoch=len(train_img)//BATCH_SIZE,
                    validation_steps=len(val_img)//BATCH_SIZE,
                    callbacks=[checkpoint, early_stop])


model.save('720unetLINESEGMENT.keras')

