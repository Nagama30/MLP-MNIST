import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

f = open('mnist.pkl', 'rb')
dictionary = pickle.load(f)
f.close()

X = dictionary['training_images']
y = dictionary['training_labels']
X_test = dictionary['test_images']
y_test = dictionary['test_labels']
num_labels = 10

print(X.shape)
print(y.shape)
print(X_test.shape)
print(y_test.shape)

# Select the first 1,000 examples for both training and testing set
num_train = 48000
num_valid = 12000 # We have a validation set
num_test = 10000

X_train = torch.FloatTensor(X[:num_train, :])
y_train = torch.LongTensor(y[:num_train])
X_valid = torch.FloatTensor(X[num_train:, :])
y_valid = torch.LongTensor(y[num_train:])
X_test = torch.FloatTensor(X_test[:num_test, :])
y_test = torch.LongTensor(y_test[:num_test])

# The grayscale of the original images ranges from 0 to 255
# We need to normalize the input data
X_train /= 255.0
X_valid /= 255.0
X_test /= 255.0

# Remember: Whatever transformations you apply to the training set, you also have to apply the same on the testing set

def save_img(img, label, file_name, num_rows = 28, num_cols = 28):
    example = np.reshape(img, (num_rows, num_cols))
    plt.matshow(example)
    plt.title("This is digit " + str(label))
    plt.savefig(file_name)

save_img(X_train[0, :], y_train[0], 'example.png')

# Convert y into one-hot vector
def convert_y(y, num_output):
    result = torch.zeros(y.size(0), num_output)
    for i in range(y.size(0)):
        result[i, y[i]] = 1
    return result

y_train = convert_y(y_train, num_output = 10)
y_valid = convert_y(y_valid, num_output = 10)
y_test = convert_y(y_test, num_output = 10)

print('Training set:')
print(X_train.size())
print(y_train.size())
print('Validation set:')
print(X_valid.size())
print(y_valid.size())
print('Testing set:')
print(X_test.size())
print(y_test.size())

# Multilayer Perceptron (MLP)
class MLP(nn.Module):
    # Constructor
    def __init__(self, num_input, hid_layers, num_output, activation_function=nn.ReLU, device='cpu'):
        super().__init__()

        self.device = device if device else torch.device("cpu")
        # Store parameters
        self.num_input = num_input
        self.num_hidden = len(hid_layers)
        self.hid_layers = hid_layers
        self.num_output = num_output
        # Store the activation function
        self.activation_function = activation_function()

        # Define input layer
        layers = [nn.Linear(self.num_input, self.hid_layers[0])]

        # Define hidden layers
        for i in range(1, len(hid_layers)):
            layers.append(nn.Linear(self.hid_layers[i-1], self.hid_layers[i]))

        # Add the output layer
        layers.append(nn.Linear(self.hid_layers[-1], self.num_output))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # This input is a mini-batch of examples
        # x is a matrix of size [batch size x number of input features]
        # hidden = torch.sigmoid(self.fc_1(x))
        # output = torch.softmax(self.fc_2(hidden), dim = 1) # We use Softmax to transform the signals of the output neurons into a probability distribution
        # return output

        # Forward pass through the layers
        for layer in self.layers[:-1]:  # For all layers except the last one
            x = self.activation_function(layer(x))
        x = self.layers[-1](x)  # Output layer without activation
        return x

# Function to choose activation function
def get_activation_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    elif name.lower() == "sigmoid":
        return nn.Sigmoid
    elif name.lower() == "tanh":
        return nn.Tanh
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU
    else:
        print("Unknown activation function. Defaulting to ReLU.")
        return nn.ReLU

# Model creation
num_input = X_train.size(1)
num_output = num_labels
initial_lr = 1e-2

# Taking input from the user
# Number of hidden layers
num_hidden_layers = int(input("Enter the number of hidden layers: "))

# Taking neurons for each hidden layer
hidden_layers = []
for i in range(num_hidden_layers):
    neurons = int(input(f"Enter the number of neurons for hidden layer {i+1}: "))
    hidden_layers.append(neurons)

# Activation function
activation_name = input("Enter the activation function (relu, sigmoid, tanh, leakyrelu): ")
activation_function = get_activation_function(activation_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is using device: {device}")
# Instantiate the MLP model with user input
model = MLP(num_input, hidden_layers, num_output, activation_function=activation_function, device=device)

# Print the model structure
print(model)

print('Done')

# If we want this model to be in GPU then
# model = model.device('cuda')
'''if torch.cuda.is_available():
    model.cuda()'''

# Optimizers: Stochastic Gradient Descent (SGD), Adam pr Adagrad
# Function to choose optimizer
def get_optimizer(name, model, initial_lr):
    if name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), initial_lr)
    elif name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), initial_lr)
    elif name.lower() == "adagrad":
        return torch.optim.Adagrad(model.parameters(), initial_lr)
    else:
        print("Unknown optimizer. Defaulting to SGD.")
        return torch.optim.SGD(model.parameters(), initial_lr)

# Taking input from the user for optimizer
optimizer_name = input("Enter the optimizer (sgd, adam, adagrad): ")
optim = get_optimizer(optimizer_name, model, initial_lr)

# optim = torch.optim.SGD(model.parameters(), lr = initial_lr)

# Stochastic Gradient Descent with Mini-Batch
num_epochs = 100
batch_size = 20
num_train_iters = X_train.size(0) // batch_size
num_valid_iters = X_valid.size(0) // batch_size
num_test_iters = X_test.size(0) // batch_size

# To store the accurancies for training and validation set
train_accuracies = []
valid_accuracies = []

best_acc = 0.0
for epoch in range(num_epochs):
    print('Epoch', epoch, '-------------------------')

	# We need to shuffle the training dataset!

    avg_loss = 0.0

    # Training
    all_train_predict = []
    all_train_truth = []
    
    # Training
    for i in range(num_train_iters):
        indices = range(i * batch_size, (i + 1) * batch_size)
        x = X_train[indices, :]
        y = y_train[indices, :]

        # Empty the gradients first
        optim.zero_grad()

        # Perform the forward pass
        predict = model(x)

		# Loss: We need to have a loss function that can compare two probability distributions. And this case, we use Cross-Entropy
        # loss = nn.CrossEntropyLoss(predict, y)
        loss = torch.nn.functional.cross_entropy(predict, y, reduction = 'mean')

        # Perform the backward pass
        loss.backward()

        # Perform SGD update
        optim.step()

        # Summing all losses
        avg_loss += loss.item()

        # Store training predictions and truths for accuracy
        all_train_predict.append(torch.argmax(predict, dim=1))
        all_train_truth.append(torch.argmax(y, dim=1))

    avg_loss /= num_train_iters
    print("Training average loss:", avg_loss)

    # Calculate training accuracy
    all_train_predict = torch.cat(all_train_predict)
    all_train_truth = torch.cat(all_train_truth)
    train_acc = torch.sum(all_train_predict == all_train_truth).item() / X_train.shape[0] * 100
    train_accuracies.append(train_acc)
    print("Training accuracy:", train_acc)
    # Evaluation on the validation set
    # torch.eval()

    all_predict = []
    all_truth = []

    with torch.no_grad():
        for i in range(num_valid_iters):
            indices = range(i * batch_size, (i + 1) * batch_size)
            x = X_valid[indices, :]
            y = y_valid[indices, :]

            predict = model(x)

            all_predict.append(torch.argmax(predict, dim = 1))
            all_truth.append(torch.argmax(y, dim = 1))

    all_predict = torch.cat(all_predict)
    all_truth = torch.cat(all_truth)

    valid_acc = torch.sum(all_predict == all_truth) / X_valid.shape[0] * 100
    valid_accuracies.append(valid_acc)
    print("Validation accuracy:", valid_acc)

    if valid_acc > best_acc:
        print("Save model to file")
        best_acc = valid_acc
        torch.save(model.state_dict(), "model.dat")
    # else:
    #   print("Early stop!")
    #   break


# After the training loop, plot and save the accuracy graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.plot(valid_accuracies, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid()
plt.savefig('accuracy_plot.png')  # Save the plot as a PNG file
#plt.show()  # Optionally, display the plot

# Evaluate one single time on the testing set
# And we only report the testing accuracy
model.load_state_dict(torch.load("model.dat")) # We load the best model (on the validation set) from file

all_predict = []
all_truth = []

with torch.no_grad():
	for i in range(num_test_iters):
		indices = range(i * batch_size, (i + 1) * batch_size)
		x = X_test[indices, :]
		y = y_test[indices, :]

		predict = model(x)

		all_predict.append(torch.argmax(predict, dim = 1))
		all_truth.append(torch.argmax(y, dim = 1))

	all_predict = torch.cat(all_predict)
	all_truth = torch.cat(all_truth)

	test_acc = torch.sum(all_predict == all_truth) / X_test.shape[0] * 100

	print("--------------------------------------------")
	print("Testing accuracy with the best model:", test_acc)

# Compute the confusion matrix
C = np.zeros((num_labels, num_labels))
for i in range(num_labels):
	indices = all_truth == i
	count = torch.sum(indices).item()
	assert count == all_predict[indices].size(0)
	arr = all_predict[indices]
	for j in range(num_labels):
		C[i, j] = torch.sum(arr == j).item() * 100.0 / count
print('Confusion matrix:\n', C)

plt.clf()
plt.matshow(C)

# Add numbers in each cell of the confusion matrix
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        plt.text(j, i, f'{C[i, j]:.1f}', ha='center', va='center', color='black')

plt.savefig('Confusion_Matrix.png')

print('Done')
