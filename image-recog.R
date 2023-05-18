library(tidyverse)
library(keras)
library(reticulate)
library(tensorflow)
library(keras)
virtualenv_create("image-recog")
install_tensorflow(envname="image-recog")
install_keras(envname="image-recog")
# base r get labels based on user defined classes
tf$constant("Hello Tensorflow!")

label_list <- dir('images/train/')
output_n <- length(label_list)
save(label_list, file="label_list.R")

# rescale and normalize our image sizes
width <- 224
height <- 224
target_size <- c(width, height)
rgb <- 3
path_train <- "images/train/"

train_data_gen <- image_data_generator(rescale = 1/255,
                                       validation_split=0.2)


train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label_list,
                                           seed = 8540)


validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen, 
                                                subset = 'validation',
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                classes = label_list,
                                                seed = 8540)

mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base)

learning_rate = 0.001;dropoutrate=0.2;n_dense=1024
  
  k_clear_session()
  
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  
  model <- model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy")

batch_size <- 32
epochs <- 6
hist <- model %>% fit(
  train_images,
  validation_data = validation_images
  )

path_test <- "/test/"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
                                          test_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F,
                                          seed = 2021)
model %>% evaluate_generator(test_images, 
                             steps = test_images$n)



lkasjd;lkfajdsawiljasdkjf;lkdsajfl





label_list <- dir('images/train/')
output_n <- length(label_list)
save(label_list, file="label_list.R")

# rescale and normalize our image sizes
width <- 224
height <- 224
target_size <- c(width, height)
rgb <- 3
path_train <- "images/train/"

train_data_gen <- image_data_generator(rescale = 1/255,
                                       validation_split=0.2)


train_images <- flow_images_from_directory(path_train,
                                           train_data_gen,
                                           subset = 'training',
                                           target_size = target_size,
                                           class_mode = "categorical",
                                           shuffle=F,
                                           classes = label_list,
                                           seed = 8540)


validation_images <- flow_images_from_directory(path_train,
                                                train_data_gen, 
                                                subset = 'validation',
                                                target_size = target_size,
                                                class_mode = "categorical",
                                                classes = label_list,
                                                seed = 8540)

mod_base <- application_xception(weights = 'imagenet', 
                                 include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base)

learning_rate = 0.001;dropoutrate=0.2;n_dense=1024

k_clear_session()

model <- keras_model_sequential() %>%
  mod_base %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = n_dense) %>%
  layer_activation("relu") %>%
  layer_dropout(dropoutrate) %>%
  layer_dense(units=output_n, activation="softmax")

model <- model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = learning_rate),
  metrics = "accuracy")

batch_size <- 32
epochs <- 6
hist <- model %>% fit(
  train_images,
  validation_data = validation_images
)

path_test <- "/test/"
test_data_gen <- image_data_generator(rescale = 1/255)
test_images <- flow_images_from_directory(path_test,
                                          test_data_gen,
                                          target_size = target_size,
                                          class_mode = "categorical",
                                          classes = label_list,
                                          shuffle = F,
                                          seed = 2021)
model %>% evaluate_generator(test_images, 
                             steps = test_images$n)
