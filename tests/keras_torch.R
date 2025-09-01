
# Setup -------------------------------------------------------------------

library(reticulate)
library(scrubwren)
library(ggplot2)

Sys.setenv(KERAS_BACKEND = "torch")

virtualenv_install(envname = "keras_torch", 
                   packages = c("keras", "torch", "torchvision", "torchaudio", "torchinfo"))

use_virtualenv("keras_torch")

keras <- import("keras", convert = FALSE)

# Linear model ------------------------------------------------------------

inputs <- keras$layers$Input(shape = tuple(1L))
output <- keras$layers$Dense(1L)(inputs)

# y ~ 1 + x + w + z
model <- keras$Model(inputs, output)

# Try different optimizer and different configs
model$compile(loss = "mean_squared_error",
              optimizer = "sgd")

model$summary()

# Train and test data -----------------------------------------------------

train_x <- matrix(rnorm(300), ncol = 1)

train_y <- matrix(1 + 2 * train_x[, 1] + rnorm(300), ncol = 1)

test_x <- matrix(rnorm(300), ncol = 1)

test_y <- matrix(1 + 2 * test_x[, 1] + rnorm(300), ncol = 1)


# Fit model ---------------------------------------------------------------

# The model should fit poorly without training
model$evaluate(train_x, train_y)

# Try different `batch_size` and `epochs`
history <- model$fit(train_x, train_y, batch_size = 32L, epochs = 50L)

# Check model -------------------------------------------------------------

# Check convergence
py_to_r(history$history) |>
  as.data.frame() |>
  ggplot(aes(1:length(loss), loss)) +
  geom_line() +
  geom_point() +
  xlab("Epoch")

# Compute MSE for train and test set
model$evaluate(train_x, train_y)
model$evaluate(test_x, test_y)

train_pred <- model$predict(train_x) |> py_to_r()
test_pred <- model$predict(test_x) |> py_to_r()

# Residual plot
result_data <- rbind(data.frame(x = train_x[, 1],
                                y = train_y[, 1],
                                fitted = train_pred,
                                resid = train_y - train_pred,
                                set = "train"),
                     data.frame(x = test_x[, 1],
                                y = test_y[, 1],
                                fitted = test_pred,
                                resid = test_y - test_pred,
                                set = "test"))

result_data |>
  ggplot() +
  geom_point(aes(fitted, resid)) +
  facet_wrap(~set)

# Check coef
py_for(p ~ model$parameters(), print(p))

# Compare to lm
lm_mod <- lm(train_y[, 1] ~ train_x[, 1])
summary(lm_mod)

# Fitted line
result_data |>
  ggplot() +
  geom_point(aes(x, y)) +
  geom_line(aes(x, fitted)) +
  geom_abline(intercept = lm_mod$coefficients[1], slope = lm_mod$coefficients[2], col = "red") +
  facet_wrap(~set)


