library(reticulate)
library(glue)

# This virtualenv is installed in ~/.virtualenvs/pytorch
virtualenv_install("pytorch", 
                   packages = c("torch", "torchvision"))
use_virtualenv("pytorch")
py_builtins <- import_builtins(convert = FALSE)

torch <- import("torch", convert = FALSE)
np <- import("numpy", convert = FALSE)


# Initializing a Tensor ---------------------------------------------------

# Tensors can be created directly from data. The data type is automatically inferred.
data <- list(c(1, 2), c(3, 4))
x_data <- torch$tensor(data)
print(x_data)

# Tensors can be created from NumPy arrays
np_array <- np$array(data)
x_np <- torch$from_numpy(np_array)

# The new tensor retains the properties (shape, datatype) of the 
# argument tensor, unless explicitly overridden.

# retains the properties of x_data
x_ones <- torch$ones_like(x_data)
glue("Ones Tensor: \n {x_ones} \n")

# overrides the datatype of x_data
x_rand <- torch$rand_like(x_data, dtype = torch$float)
glue("Random Tensor: \n {x_rand} \n")


# In the functions below, `shape` determines the dimensionality of the output tensor.
shape <- c(2L, 3L)
rand_tensor <- torch$rand(shape)
ones_tensor <- torch$ones(shape)
zeros_tensor <- torch$zeros(shape)

glue("Random Tensor: \n {rand_tensor} \n")
glue("Ones Tensor: \n {ones_tensor} \n")
glue("Zeros Tensor: \n {zeros_tensor}")


# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor <- torch$rand(3L, 4L)

glue("Shape of tensor: {tensor$shape}")
glue("Datatype of tensor: {tensor$dtype}")
glue("Device tensor is stored on: {tensor$device}")


# Operations on Tensors ---------------------------------------------------

# By default, tensors are created on the CPU.
# We need to explicitly move tensors to the accelerator using .to method 
# (after checking for accelerator availability). 
# Keep in mind that copying large tensors across devices can be expensive 
# in terms of time and memory!

# We move our tensor to the current accelerator if available
if (torch$accelerator$is_available() |> py_to_r())
  tensor <- tensor$to(torch$accelerator$current_accelerator())

# Standard array like indexing and slicing:
tensor <- torch$arange(1L, 17L)$reshape(c(4L, 4L))
print(tensor)

glue("First row: {tensor[0]}")
glue("First column: {tensor[, 0]}")
glue("Last column: {tensor[, -1]}")

# Set the second column to zero
tensor[, 1] <- 0
print(tensor)

# You can use torch$cat to concatenate a sequence of tensors 
# along an existing dimension.
t1 <- torch$cat(list(tensor, tensor, tensor), dim = 0L)
print(t1)
glue("Shape of t1: {t1$shape}")

# You can use torch$stack to concatenate a sequence of tensors 
# along a new dimension (the new dimension will be inserted).
t2 <- torch$stack(list(tensor, tensor, tensor), dim = 0L)
print(t2)
glue("Shape of t2: {t2$shape}")

# Arithmetic operations

# Matrix transpose
tensor$t()

# Matrix multiplication
y1 <- tensor %*% tensor$t()
y2 <- tensor$matmul(tensor$t())
y3 <- tensor$new_empty(c(4L, 4L))
torch$matmul(tensor, tensor$t(), out = y3)

# Matrix element-wise product
z1 <- tensor * tensor
z2 <- tensor$mul(tensor)
z3 <- tensor$new_empty(c(4L, 4L))
torch$mul(tensor, tensor, out = z3)

# If you have a one-element tensor, for example by aggregating all values of a 
# tensor into one value, you can convert it to a Python numerical value using item()
agg_item <- tensor$sum()$item()
glue("{agg_item} {py_builtins$type(agg_item)}")

# Operations that store the result into the operand are called in-place. 
# They are denoted by a _ suffix
print(tensor)
tensor$add_(5L)
print(tensor)


# Bridge with NumPy -------------------------------------------------------

# Tensors on the CPU and NumPy arrays can share their underlying memory 
# locations, and changing one will change the other.

# Tensor to NumPy array
t <- torch$ones(5L)
glue("t: {t}")
n <- t$numpy()
glue("n: {n}")

# A change in the tensor reflects in the NumPy array.
t$add_(1)
glue("t: {t}")
glue("n: {n}")

# NumPy array to Tensor
n <- np$ones(5L)
t <- torch$from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np$add(n, 1, out = n)
glue("t: {t}")
glue("n: {n}")
