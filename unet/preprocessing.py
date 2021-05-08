
from keras.preprocessing import image


def create_image_mask_generator(X_train, Y_train, batch_size, seed):
  # Creating the training Image and Mask generator
  image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                           height_shift_range=0.2, fill_mode='reflect')
  mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                          height_shift_range=0.2, fill_mode='reflect')

  # Keep the same seed for image and mask generators so they fit together

  image_datagen.fit(X_train[:int(X_train.shape[0]*0.9)], augment=True, seed=seed)
  mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.9)], augment=True, seed=seed)

  x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.9)],batch_size=batch_size, shuffle=True, seed=seed)
  y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.9)],batch_size=batch_size, shuffle=True, seed=seed)


  # Creating the validation Image and Mask generator
  image_datagen_val = image.ImageDataGenerator()
  mask_datagen_val = image.ImageDataGenerator()

  image_datagen_val.fit(X_train[int(X_train.shape[0]*0.9):], augment=True, seed=seed)
  mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.9):], augment=True, seed=seed)

  x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.9):],batch_size=batch_size, shuffle=True, seed=seed)
  y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.9):],batch_size=batch_size, shuffle=True, seed=seed)

  #creating a training and validation generator that generate masks and images
  train_generator = zip(x, y)
  val_generator = zip(x_val, y_val)

  return train_generator, val_generator

