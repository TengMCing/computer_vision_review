library(reticulate)
library(cli)
library(ggplot2)
options(cli.progress_show_after = 0)
options(cli.width = 200)

# This virtualenv is installed in ~/.virtualenvs/pytorch
use_virtualenv("pytorch")
py_builtins <- import_builtins(convert = FALSE)

# Check if torch works
torch <- import("torch", convert = FALSE)
x <- torch$rand(5L, 3L)
print(x)

# Working with data
torchvision <- import("torchvision", convert = FALSE)


# Load Data ---------------------------------------------------------------

# datasets stores data samples and corresponding labels
datasets <- torchvision$datasets

# Download and load the dataset
fashion_mnist_train <- datasets$FashionMNIST(root = here::here("tests/fashion_mnist"),
                                             train = TRUE,
                                             download = TRUE,
                                             transform = torchvision$transforms$ToTensor())

fashion_mnist_test <- datasets$FashionMNIST(root = here::here("tests/fashion_mnist"),
                                            train = FALSE,
                                            download = TRUE,
                                            transform = torchvision$transforms$ToTensor())

# DataLoader provides a iterable around datasets
DataLoader <- torch$utils$data$DataLoader

# Load samples from datasets
batch_size <- 64L
fashion_mnist_train_loader <- DataLoader(fashion_mnist_train, batch_size = batch_size, shuffle = T)
fashion_mnist_test_loader <- DataLoader(fashion_mnist_test, batch_size = batch_size, shuffle = F)

# DataLoader only implements the __iter__ method but not the __next__ method,
# so we need to convert it to a legitimate iterator.
# Try to get a batch from the test data
item <- iter_next(as_iterator(fashion_mnist_test_loader))
X <- item[0]
y <- item[1]
cli_alert_info("{.field Shape of X [N, C, H, W]:} {.val {X$shape}}, {.field Type of X:} {.val {X$dtype}}")
cli_alert_info("{.field Shape of y:} {.val {y$shape}}, {.field Type of y:} {.val {y$dtype}}")


# Enable GPU if available
if (py_to_r(torch$accelerator$is_available())) {
  device <- torch$accelerator$current_accelerator()$type
} else {
  device <- "cpu"
}
cli_alert_info("Using {.field device:} {.val {device}}")


# Define Model ------------------------------------------------------------

# nn is the neural network module
nn <- torch$nn

# Subclassing a neural network, inherited from nn.Module
NeuralNetwork <- PyClass(classname = "NeuralNetwork", inherit = nn$Module,
                         list(
                           `__init__` = function(self) {
                             # Use the super(type, object_or_type) advanced 
                             # syntax because we are calling from R and 
                             # Python can not automatically detect self.
                             # `__class__` is defined. It is a Python class
                             # existed in the current R scope.
                             py_builtins$super(`__class__`, self)$`__init__`()
                             
                             # The flatten preprocessor helps flatten the pixels into a vector
                             self$flatten <- nn$Flatten()
                             
                             # The sequential network
                             self$linear_relu_stack <- nn$Sequential(
                               nn$Linear(28L * 28L, 512L),
                               nn$ReLU(),
                               nn$Linear(512L, 512L),
                               nn$ReLU(),
                               nn$Linear(512L, 10L)
                             )
                             
                             return(py_none())
                           },
                           
                           # The forward pass, the input will be flattened, then passed to the linear units
                           forward = function(self, x) {
                             x <- self$flatten(x)
                             logits <- self$linear_relu_stack(x)
                           }
                         ))
          

# Init the model and send it to GPU if available
model <- NeuralNetwork()$to(device)

# Use torchinfo to get a Keras-like summary table
torchinfo <- import("torchinfo", convert = FALSE)
torchinfo$summary(model, verbose = 0L)


# Loss and Optimizer ------------------------------------------------------

loss_fn <- nn$CrossEntropyLoss()
optimizer <- torch$optim$SGD(model$parameters(),
                             lr = 1e-3)


# Training Step -----------------------------------------------------------

train <- function(dataloader, model, loss_fn, optimizer) {
  
  # Create a new iterator from the iterable dataloader
  iter <- as_iterator(dataloader)
  batches <- py_to_r(py_builtins$len(dataloader))
  vec_loss <- c()
  vec_correct <- c()
  
  # Setup progress bar
  cli_progress_bar(total = batches,
                   format = "{pb_spin} {col_red('Training')} {.val {pb_current}}/{.val {pb_total}} {pb_bar} | {.field ETA:} {pb_eta} | {.field loss:} {sprintf(mean(vec_loss), fmt = '%#.4f')} | {.field accuracy:} {sprintf(mean(vec_correct), fmt = '%#.4f')}",
                   format_done = "{col_green(symbol$tick)} {col_green('Trained ')} {.val {pb_current}}/{.val {pb_total}} | {sprintf(pb_elapsed_raw, fmt = '%#.2f')}s - {sprintf(pb_elapsed_raw/batches, fmt = '%#.4f')}s/step | {.field loss:} {sprintf(mean(vec_loss), fmt = '%#.4f')} | {.field accuracy:} {sprintf(mean(vec_correct), fmt = '%#.4f')}",
                   clear = FALSE)
  
  # Set the model in training mode
  model$train()
  
  while (TRUE) {
    # Get one batch and send them to GPU if available
    item <- iter_next(iter, completed = quote(StopIteration))
    if (identical(item, quote(StopIteration))) break
    X <- item[0]$to(device)
    y <- item[1]$to(device)
    
    # Compute model prediction and cross entropy
    pred <- model(X)
    loss <- loss_fn(pred, y)
    
    # Perform backpropagation
    # Compute gradient
    loss$backward()
    # Update by one step
    optimizer$step()
    # Set gradient to zero otherwise gradient will be accumulated (i.e. +=) 
    optimizer$zero_grad()
    
    # Record the loss and boolean vectors
    vec_loss <- c(vec_loss, py_to_r(loss$item()))
    vec_correct <- c(vec_correct, py_to_r((pred$argmax(1L) == y)$tolist()))
    
    cli_progress_update()
  }
  
  return(data.frame(loss = mean(vec_loss), accuracy = mean(vec_correct)))
}


# Testing Step ------------------------------------------------------------

test <- function(dataloader, model, loss_fn) {
  
  # Create a new iterator from the iterable dataloader
  iter <- as_iterator(dataloader)
  batches <- py_to_r(py_builtins$len(dataloader))
  vec_loss <- c()
  vec_correct <- c()
  
  # Setup progress bar
  cli_progress_bar(total = batches,
                   format = "{pb_spin} {col_red('Testing ')} {.val {pb_current}}/{.val {pb_total}} {pb_bar} | {.field ETA:} {pb_eta} | {.field val_loss:} {sprintf(mean(vec_loss), fmt = '%#.4f')} | {.field val_accuracy:} {sprintf(mean(vec_correct), fmt = '%#.4f')}",
                   format_done = "{col_green(symbol$tick)} {col_green('Tested  ')} {.val {pb_current}}/{.val {pb_total}} | {sprintf(pb_elapsed_raw, fmt = '%#.2f')}s - {sprintf(pb_elapsed_raw/batches, fmt = '%#.4f')}s/step | {.field val_loss:} {sprintf(mean(vec_loss), fmt = '%#.4f')} | {.field val_accuracy:} {sprintf(mean(vec_correct), fmt = '%#.4f')}",
                   clear = FALSE)
  
  # Set the model in evaluation mode
  model$eval()
  
  # Disable gradient computation, we can not call tensor.backward(), but it
  # reduces memory consumption.
  with(torch$no_grad(), {
    while (TRUE) {
      
      # Get one batch and send them to GPU if available
      item <- iter_next(iter, completed = quote(StopIteration))
      if (identical(item, quote(StopIteration))) break
      X <- item[0]$to(device)
      y <- item[1]$to(device)
      
      # Compute model prediction and cross entropy
      pred <- model(X)
      loss <- loss_fn(pred, y)
      
      # Record the loss and boolean vectors
      vec_loss <- c(vec_loss, py_to_r(loss$item()))
      vec_correct <- c(vec_correct, py_to_r((pred$argmax(1L) == y)$tolist()))
      
      cli_progress_update()
    }
  })
  
  return(data.frame(loss = mean(vec_loss), accuracy = mean(vec_correct)))
}


# Epochs ------------------------------------------------------------------

epochs <- 50L
train_history <- data.frame()
test_history <- data.frame()
for (i in 1:epochs) {
  # Log for epochs
  cli_alert_info("{col_yellow('Epoch')} {col_yellow(i)}/{col_yellow(epochs)}")
  train_history <- rbind(train_history, train(fashion_mnist_train_loader, model, loss_fn, optimizer))
  test_history <- rbind(test_history, test(fashion_mnist_test_loader, model, loss_fn))
}

# Model History Plot ------------------------------------------------------

train_history$epoch <- 1:epochs
test_history$epoch <- 1:epochs
train_history$set <- "train"
test_history$set <- "test"

rbind(train_history, test_history) |>
  tidyr::pivot_longer(c(loss, accuracy), names_to = "metric", values_to = "value") |>
  ggplot() +
  geom_point(aes(epoch, value, col = set)) +
  geom_line(aes(epoch, value, col = set)) +
  facet_wrap(~metric, scales = "free_y")

# Example -----------------------------------------------------------------

# Image of a shoe
example_img <- as.data.frame(py_to_r(X[0][0]$numpy()))
names(example_img) <- 1L:28L
example_img$rows <- rev(1L:28L)
example_img <- tidyr::pivot_longer(example_img, 1L:28L, names_to = "cols", values_to = "intensity")
example_img$cols <- as.integer(example_img$cols)

ggplot(example_img) +
  geom_raster(aes(cols, rows, fill = intensity)) +
  scale_fill_gradient(low = "black", high = "white") +
  coord_equal() +
  theme_void()

# Class definition
classes <- c(
  "T-shirt/top",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot")

# Predict the image on the appropriate device 
pred <- model(X[0]$to(device)$unsqueeze(0L))
cli_alert_info("Model prediction: {classes[pred[0]$argmax()$item() + 1]}")


# Save Model --------------------------------------------------------------

torch$save(model$state_dict(), here::here("tests/model/fashion_mnist_nn.pth"))
cli_alert_success("Model saved to {.path tests/model/fashion_mnist_nn.pth}")

# Load Model --------------------------------------------------------------

# This is the recommend way to save and load model
model <- NeuralNetwork()$to(device)
model$load_state_dict(torch.load(here::here("tests/model/fashion_mnist_nn.pth"), 
                                 weights_only = TRUE))



