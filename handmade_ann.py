import numpy as np

# ------------------------------------------------------------
# TRAINING DATA SETUP
# ------------------------------------------------------------

# Fix the random-number generator so the same training data and
# starting weights are produced each time the program is run.
# This makes testing and debugging repeatable.
np.random.seed(42)

# Number of arithmetic sequences that will be generated for training.
NUM_EXAMPLES = 50_000


# Generate every combination of starting values from 1 through 500
# and step sizes from 1 through 100.
#
# Using a smaller starting-value range and a wider step range prevents
# the absolute starting number from dominating the training data.
#
# 500 starting values * 100 step sizes = 50,000 unique examples.
starts = np.repeat(
    np.arange(1, 501),
    100
)

steps = np.tile(
    np.arange(1, 101),
    500
)

# Create the input matrix X.
#
# Each row contains three numbers from an arithmetic sequence:
#
# starting number
# starting number + step
# starting number + two steps
#
# Example:
# start = 21
# step = 21
#
# X row:
# [21, 42, 63]
#
# column_stack combines the three arrays into a matrix with
# 100,000 rows and 3 columns.
#
# astype(float) converts the integer values to floating-point
# values because neural-network calculations use decimal values.
X = np.column_stack([
    starts,
    starts + steps,
    starts + (2 * steps)
]).astype(float)

# Create y, the expected/correct output for every sequence.
#
# The fourth number is:
#
# start + (3 * step)
#
# Example:
# [21, 42, 63] -> 84
#
# reshape(-1, 1) converts y into a matrix containing one output
# column. NumPy automatically determines the required number of rows.
y = (
    starts + (3 * steps)
).reshape(-1, 1).astype(float)

# ------------------------------------------------------------
# DATA NORMALIZATION
# ------------------------------------------------------------

# Calculate the mean and standard deviation for each input column.
# These values will be used to center and normalize the training data.
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

# Calculate the mean and standard deviation of the expected outputs.
y_mean = np.mean(y)
y_std = np.std(y)

# Standardize each input column so that it has approximately
# a mean of 0 and a standard deviation of 1.
#
# This prevents the large starting values from overwhelming the
# smaller differences between numbers in each sequence.
X_scaled = (X - X_mean) / X_std

# Standardize the expected output using the same idea.
y_scaled = (y - y_mean) / y_std
# ------------------------------------------------------------
# ARTIFICIAL NEURAL NETWORK
# ------------------------------------------------------------

class NeuralNetwork:

    def __init__(self):

        # The ANN receives three numbers from each sequence.
        self.input_size = 3

        # Number of neurons in the hidden layer.
        self.hidden_size = 32

        # The network produces one predicted next number.
        self.output_size = 1


        # ----------------------------------------------------
        # INPUT -> HIDDEN LAYER PARAMETERS
        # ----------------------------------------------------

        # W1 contains the weights connecting the 3 input neurons
        # to the 32 hidden neurons.
        #
        # Shape:
        # (3, 32)
        #
        # randn() generates random starting values centered near zero.
        # Multiplying by 0.1 keeps the initial weights small.
        self.W1 = (
            np.random.randn(self.input_size, self.hidden_size) * 0.1
        )

        # b1 contains one bias for each of the 32 hidden neurons.
        # All biases initially start at zero.
        #
        # Shape:
        # (1, 32)
        self.b1 = np.zeros((1, self.hidden_size))


        # ----------------------------------------------------
        # HIDDEN -> OUTPUT LAYER PARAMETERS
        # ----------------------------------------------------

        # W2 contains the weights connecting the 32 hidden neurons
        # to the single output neuron.
        #
        # Shape:
        # (32, 1)
        self.W2 = (
            np.random.randn(self.hidden_size, self.output_size) * 0.1
        )

        # b2 is the bias for the single output neuron.
        #
        # Shape:
        # (1, 1)
        self.b2 = np.zeros((1, self.output_size))


        print("Neural network created.")
        print("Input neurons:", self.input_size)
        print("Hidden neurons:", self.hidden_size)
        print("Output neurons:", self.output_size)


    # --------------------------------------------------------
    # ACTIVATION FUNCTION
    # --------------------------------------------------------

    def relu(self, x):

        # ReLU = Rectified Linear Unit.
        #
        # Positive values remain unchanged.
        # Negative values become zero.
        #
        # Example:
        # [-2, 3, -1, 5]
        #
        # becomes:
        # [0, 3, 0, 5]
        return np.maximum(0, x)


    def relu_derivative(self, x):

        # The derivative of ReLU is used during backpropagation.
        #
        # If the original value was positive, return 1.
        # If it was zero or negative, return 0.
        #
        # This determines whether an error gradient can pass
        # backward through a hidden neuron.
        return (x > 0).astype(float)


    # --------------------------------------------------------
    # FEEDFORWARD
    # --------------------------------------------------------

    def forward(self, X):

        # Multiply the input matrix by W1 and add the hidden biases.
        #
        # X:
        # (number of examples, 3)
        #
        # W1:
        # (3, 32)
        #
        # Result:
        # (number of examples, 32)
        self.hidden_input = np.dot(X, self.W1) + self.b1


        # Apply the ReLU activation function to every hidden neuron.
        self.hidden_output = self.relu(self.hidden_input)


        # Multiply the 32 hidden-neuron outputs by W2 and add
        # the output bias.
        #
        # This reduces the 32 hidden values to one prediction.
        self.final_input = (
            np.dot(self.hidden_output, self.W2) + self.b2
        )


        # Return the ANN's predicted output.
        return self.final_input


    # --------------------------------------------------------
    # LOSS FUNCTION
    # --------------------------------------------------------

    def mse_loss(self, predicted, actual):

        # Mean Squared Error (MSE) measures how far the network's
        # predictions are from the correct answers.
        #
        # 1. predicted - actual = error
        # 2. square the error
        # 3. calculate the average
        return np.mean((predicted - actual) ** 2)


    def mse_derivative(self, predicted, actual):

        # Calculate the derivative/gradient of the MSE loss.
        #
        # This tells backpropagation both:
        #
        # - the direction of the error
        # - the size of the error
        #
        # actual.size divides the gradient across all output values
        # being processed.
        return 2 * (predicted - actual) / actual.size


    # --------------------------------------------------------
    # OUTPUT-LAYER BACKPROPAGATION
    # --------------------------------------------------------

    def output_weight_gradient(self, loss_gradient):

        # Determine how much each W2 weight contributed to the error.
        #
        # hidden_output.T changes:
        #
        # (examples, 32)
        #
        # into:
        #
        # (32, examples)
        #
        # The resulting gradient has the same shape as W2:
        #
        # (32, 1)
        dW2 = np.dot(
            self.hidden_output.T,
            loss_gradient
        )

        return dW2


    def output_bias_gradient(self, loss_gradient):

        # Calculate the gradient for the output bias b2.
        #
        # The gradients are summed across all training examples.
        #
        # keepdims=True keeps the result in matrix form:
        #
        # (1, 1)
        db2 = np.sum(
            loss_gradient,
            axis=0,
            keepdims=True
        )

        return db2


    # --------------------------------------------------------
    # HIDDEN-LAYER BACKPROPAGATION
    # --------------------------------------------------------

    def hidden_error_gradient(self, loss_gradient):

        # Send the output error backward through W2.
        #
        # W2.T reverses the orientation of W2 so the single output
        # error can be distributed among the 32 hidden neurons.
        hidden_gradient = np.dot(
            loss_gradient,
            self.W2.T
        )

        return hidden_gradient


    def apply_relu_gradient(self, hidden_gradient):

        # Apply the derivative of ReLU to the hidden-layer gradient.
        #
        # Hidden neurons that ReLU turned off receive a gradient of 0.
        # Active neurons allow their gradient to continue backward.
        return (
            hidden_gradient
            * self.relu_derivative(self.hidden_input)
        )


    # --------------------------------------------------------
    # INPUT-LAYER BACKPROPAGATION
    # --------------------------------------------------------

    def input_weight_gradient(self, X, relu_hidden_gradient):

        # Calculate the gradient for W1.
        #
        # X.T changes the input matrix so it can be multiplied by
        # the hidden-layer gradients.
        #
        # Resulting shape:
        #
        # (3, 32)
        #
        # This matches W1.
        dW1 = np.dot(
            X.T,
            relu_hidden_gradient
        )

        return dW1


    def input_bias_gradient(self, relu_hidden_gradient):

        # Calculate the gradients for the 32 hidden-layer biases.
        #
        # Resulting shape:
        #
        # (1, 32)
        #
        # This matches b1.
        db1 = np.sum(
            relu_hidden_gradient,
            axis=0,
            keepdims=True
        )

        return db1


    # --------------------------------------------------------
    # GRADIENT DESCENT
    # --------------------------------------------------------

    def update_parameters(
        self,
        dW1,
        db1,
        dW2,
        db2,
        learning_rate
    ):

        # Gradient descent moves each parameter in the opposite
        # direction of its gradient.
        #
        # The learning rate controls the size of each adjustment.

        # Update input -> hidden weights.
        self.W1 -= learning_rate * dW1

        # Update hidden-layer biases.
        self.b1 -= learning_rate * db1

        # Update hidden -> output weights.
        self.W2 -= learning_rate * dW2

        # Update output bias.
        self.b2 -= learning_rate * db2


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def train(
        self,
        X,
        y,
        iterations=1000,
        learning_rate=0.01
    ):

        # Repeat the complete learning process for the requested
        # number of iterations.
        for i in range(iterations):

            # -------------------------
            # 1. FEEDFORWARD
            # -------------------------

            # Use the current weights and biases to make predictions.
            prediction = self.forward(X)


            # -------------------------
            # 2. CALCULATE LOSS
            # -------------------------

            # Measure how far the predictions are from the
            # expected answers.
            loss = self.mse_loss(
                prediction,
                y
            )


            # -------------------------
            # 3. LOSS GRADIENT
            # -------------------------

            # Calculate the starting gradient for backpropagation.
            loss_gradient = self.mse_derivative(
                prediction,
                y
            )


            # -------------------------
            # 4. OUTPUT LAYER
            # -------------------------

            # Calculate gradients for W2 and b2.
            dW2 = self.output_weight_gradient(
                loss_gradient
            )

            db2 = self.output_bias_gradient(
                loss_gradient
            )


            # -------------------------
            # 5. HIDDEN LAYER
            # -------------------------

            # Send the error backward from the output layer
            # into the hidden layer.
            hidden_gradient = self.hidden_error_gradient(
                loss_gradient
            )


            # Apply the derivative of ReLU.
            relu_hidden_gradient = self.apply_relu_gradient(
                hidden_gradient
            )


            # -------------------------
            # 6. INPUT -> HIDDEN LAYER
            # -------------------------

            # Calculate gradients for W1 and b1.
            dW1 = self.input_weight_gradient(
                X,
                relu_hidden_gradient
            )

            db1 = self.input_bias_gradient(
                relu_hidden_gradient
            )


            # -------------------------
            # 7. UPDATE PARAMETERS
            # -------------------------

            # Apply gradient descent to all weights and biases.
            self.update_parameters(
                dW1,
                db1,
                dW2,
                db2,
                learning_rate
            )


            # Display progress every 100 iterations.
            #
            # i + 1 is used so the display starts at iteration 1
            # instead of iteration 0.
            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"Iteration {i + 1}: "
                    f"Loss = {loss:.10f}"
                )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, values):

        # Convert the user's three input values into a NumPy array.
        # reshape(1, 3) formats the values as one row with
        # three columns so they match the ANN's expected input shape.
        #
        # Example:
        # [200, 202, 204]
        #
        # becomes:
        # [[200.0, 202.0, 204.0]]
        values = np.array(
            values,
            dtype=float
        ).reshape(1, 3)


        # Normalize the new input using the SAME mean and standard
        # deviation values that were calculated from the training data.
        #
        # This is important because the ANN was trained using
        # standardized inputs rather than the original raw numbers.
        scaled_values = (
            values - X_mean
        ) / X_std


        # Send the normalized values through the trained neural network
        # to generate a normalized prediction.
        scaled_prediction = self.forward(
            scaled_values
        )


        # Convert the normalized prediction back to the original
        # number range using the training-output mean and standard
        # deviation.
        prediction = (
            scaled_prediction * y_std
        ) + y_mean


        # prediction is stored as a 2D NumPy array with shape (1, 1).
        # [0][0] extracts the single numeric prediction from it.
        return prediction[0][0]
    
# ------------------------------------------------------------
# CREATE AND TRAIN THE ANN
# ------------------------------------------------------------

print("Input shape:", X.shape)
print("Output shape:", y.shape)

print("\nCreating neural network...")

ann = NeuralNetwork()

print("\nTraining neural network...")

# Train for at least 1,000 iterations as required by the assignment.
ann.train(
    X_scaled,
    y_scaled,
    iterations=6000,
    learning_rate=0.05
)

print("\nTraining complete.")

# ------------------------------------------------------------
# PREDICTION TEST
# ------------------------------------------------------------

# ------------------------------------------------------------
# PREDICTION TESTS
# ------------------------------------------------------------

# Test the trained ANN against several arithmetic sequences
# with different starting values and step sizes.
test_sequences = [
    ([2, 4, 6], 8),
    ([3, 6, 9], 12),
    ([10, 11, 12], 13),
    ([30, 60, 90], 120),
    ([50, 75, 100], 125),
    ([120, 123, 126], 129),
    ([200, 202, 204], 206),
]

print("\nPrediction tests:")

for values, expected in test_sequences:

    prediction = ann.predict(values)

    absolute_error = abs(prediction - expected)

    print("\nInput values:", values)
    print("Expected next number:", expected)
    print("ANN prediction:", prediction)
    print("Rounded prediction:", round(prediction))
    print("Absolute error:", absolute_error)

# ------------------------------------------------------------
# INTERACTIVE USER PREDICTION
# ------------------------------------------------------------

# Ask the user to enter three numbers separated by spaces.
# Example:
# 2 4 6
user_input = input(
    "\nEnter three numbers separated by spaces: "
)

# Split the typed text into separate values and convert each
# one to a floating-point number.
user_values = [
    float(value)
    for value in user_input.split()
]

# Make sure exactly three values were entered.
if len(user_values) == 3:

    # Send the user's three values through the trained ANN.
    user_prediction = ann.predict(user_values)

    print("\nANN Prediction")
    print("Input values:", user_values)
    print("Predicted next number:", user_prediction)
    print("Rounded prediction:", round(user_prediction))

else:

    # Display an error message if the user enters more or fewer
    # than three numbers.
    print("\nPlease enter exactly three numbers.")

# ------------------------------------------------------------
# DEVELOPMENT / DIAGNOSTIC TESTING CODE
# ------------------------------------------------------------
#
# This is the testing code used while constructing the ANN.
# It is intentionally left in the program for reference but
# commented out so it does not execute during normal training.
#
# ------------------------------------------------------------


# print("Input shape:", X.shape)
# print("Output shape:", y.shape)
#
#
# print("\nFirst 5 original training examples:")
#
# for i in range(5):
#     print(f"{X[i]} -> {y[i][0]}")
#
#
# print("\nFirst 5 scaled training examples:")
#
# for i in range(5):
#     print(f"{X_scaled[i]} -> {y_scaled[i][0]}")
#
#
# ann = NeuralNetwork()
#
#
# # Run only the first training example through the ANN.
# prediction = ann.forward(X_scaled[:1])
#
# print("\nPrediction for first training example:")
# print(prediction)
#
# print("Expected scaled output:")
# print(y_scaled[:1])
#
#
# # Calculate the MSE loss for the first example.
# loss = ann.mse_loss(
#     prediction,
#     y_scaled[:1]
# )
#
# print("Loss:")
# print(loss)
#
#
# # Calculate the derivative of the loss.
# loss_gradient = ann.mse_derivative(
#     prediction,
#     y_scaled[:1]
# )
#
# print("Loss gradient:")
# print(loss_gradient)
#
#
# # Calculate the W2 gradient.
# dW2 = ann.output_weight_gradient(
#     loss_gradient
# )
#
# print("W2 gradient shape:")
# print(dW2.shape)
#
# print("First 5 W2 gradients:")
# print(dW2[:5])
#
#
# # Calculate the b2 gradient.
# db2 = ann.output_bias_gradient(
#     loss_gradient
# )
#
# print("b2 gradient:")
# print(db2)
#
# print("b2 gradient shape:")
# print(db2.shape)
#
#
# # Send the error backward into the hidden layer.
# hidden_gradient = ann.hidden_error_gradient(
#     loss_gradient
# )
#
# print("Hidden gradient shape:")
# print(hidden_gradient.shape)
#
# print("First 5 hidden gradients:")
# print(hidden_gradient[0][:5])
#
#
# # Apply the ReLU derivative.
# relu_hidden_gradient = ann.apply_relu_gradient(
#     hidden_gradient
# )
#
# print("Hidden gradient after ReLU:")
# print(relu_hidden_gradient[0][:5])
#
#
# # Calculate W1 and b1 gradients.
# dW1 = ann.input_weight_gradient(
#     X_scaled[:1],
#     relu_hidden_gradient
# )
#
# db1 = ann.input_bias_gradient(
#     relu_hidden_gradient
# )
#
# print("W1 gradient shape:")
# print(dW1.shape)
#
# print("b1 gradient shape:")
# print(db1.shape)
#
# print("First 5 W1 gradients for input 1:")
# print(dW1[0][:5])
#
#
# # Test a single gradient-descent update.
# learning_rate = 0.1
#
# ann.update_parameters(
#     dW1,
#     db1,
#     dW2,
#     db2,
#     learning_rate
# )
#
#
# # Run the same example through the ANN again after updating
# # the weights and biases.
# new_prediction = ann.forward(
#     X_scaled[:1]
# )
#
# new_loss = ann.mse_loss(
#     new_prediction,
#     y_scaled[:1]
# )
#
# print("\nAfter one gradient descent update:")
#
# print("New prediction:")
# print(new_prediction)
#
# print("New loss:")
# print(new_loss)