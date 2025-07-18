library(reticulate)
library(cli)

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
  
  iter <- as_iterator(dataloader)
  cli_progress_bar("Training", 
                   total = py_builtins$len(fashion_mnist_train_loader),
                   clear = FALSE)
  
  # Set the model in training mode
  model$train()
  
  batch <- 0L
  while (TRUE) {
    batch <- batch + 1L
    
    # Get one batch and send them to GPU if available
    item <- iter_next(fashion_mnist_train_loader_iter, 
                      completed = quote(StopIteration))
    if (item == quote(StopIteration)) break
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
    
    cli_progress_update()
  }
  
}

# Define 