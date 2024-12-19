# MLP

- The MLP class take in a specified number of input features (num_input), a list of hidden layers 
(hid_layers), the number of output features (num_output), and an optional activation function 
(defaulting to ReLU). In the constructor (__init__), the input layer is first defined using nn.Linear, 
and then hidden layers are iteratively added. The layers are stored in a nn.ModuleList, and the 
chosen activation function is applied between hidden layers.

- The function get_activation_function(name) is defined to return the appropriate activation function 
(ReLU, Sigmoid, Tanh, or LeakyReLU) based on a string input from the user. The forward method 
loops through all layers except the last one, applying the specified activation function at each step. 
- The user is prompted to specify the number of hidden layers, the number of neurons in each hidden 
layer, and the activation function to be used. Once all inputs are collected, the MLP model is 
instantiated with the provided parameters.

- The function get_optimizer that selects an optimizer for training a neural network based on user 
input. It supports Stochastic Gradient Descent (SGD), Adam, and Adagrad optimizers from 
PyTorch, each initialized with the model's parameters and a specified learning rate (initial_lr). If 
an unrecognized optimizer name is provided, the function defaults to using SGD. The user is 
prompted to input their choice of optimizer, which is then applied to the model.
 
- In the program, it checks if a cuda capable GPU is available and sets the device to "cuda". 
Otherwise, it defaults to the CPU. We create an instance of the MLP (Multilayer Perceptron) model 
with input parameters such as the number of input features, hidden layers, number of output labels, 
and activation function and the device.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Model is using device: {device}") 
Instantiate the MLP model with user input 
Model = MLP(num_input, hidden_layers, num_output, 
activation_function=activation_function, device=device) 


A. MNIST dataset: For a sample execution, I have used the parameters below for model training. 
- Parameters: 
Number of hidden layers: 2 
Neurons in each layer sequentially: 200, 200 
Activation function: relu 
Optimization method: sgd 
MLP( 
(activation_function): ReLU() 
(layers): ModuleList( 
(0): Linear(in_features=784, out_features=200, bias=True) 
(1): Linear(in_features=200, out_features=200, bias=True) 
(2): Linear(in_features=200, out_features=10, bias=True) 
) 
) 

a) Training and validation accuracy after each epoch-  
Below graph shows training and validation accuracy after each epoch.
At epoch 0: 
Training accuracy is 75.93 and validation accuracy is 89.11 
At epoch 99: 
Training accuracy is 100 and validation accuracy is 97.62. 

![image](https://github.com/user-attachments/assets/570a6ad6-f3c5-41da-8caf-ae83012a575e)

b) Testing accuracy- 
Testing accuracy with the best model is 97.90 
c) Confusion matrix- 
Confusion matrix is diagonally dominant. So, all the images (almost) are classified correctly.

![image](https://github.com/user-attachments/assets/ab4c4b2e-3cf9-40c4-beef-0ffcea790c02)


