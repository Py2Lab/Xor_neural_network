import numpy as np 
import itertools
#np.random.seed(0)

class XorNeuralNetork:
    
    def __init__(self, number_variable):
        self.number_variable = number_variable
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def generate_input(self, number_varible):
        l = [0, 1]
        input_value =  [list(i) for i in itertools.product(l, repeat=number_varible)]
        return np.array(input_value)

    def genereate_expected_output(self, number_variable):
        expected_output = [ [1] for i in range(1,  2**number_variable - 1)]
        expected_output.insert(0,[0])
        expected_output.append([0])
        return np.array(expected_output)

    def random_weights_bias(self, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons):
        #Random weights and bias initialization
        hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
        hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
        output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
        output_bias = np.random.uniform(size=(1,outputLayerNeurons))

        print("Initial hidden weights, biases:{0},{1} ".format(*hidden_weights, *hidden_bias))
        print("Initial output weights, biases:{0},{1} ".format(*output_weights, *output_bias))
        
        return hidden_weights, hidden_bias, output_weights, output_bias


    def training_algorithm(self, epochs, lr):
        #Training algorithm
        random_weights = self.random_weights_bias(self.number_variable, self.number_variable, 1)
        hidden_weights, hidden_bias, output_weights, output_bias = random_weights
        inputs, expected_output = self.generate_input(self.number_variable), self.genereate_expected_output(self.number_variable)
        for _ in range(epochs):
            #Forward Propagation
            hidden_layer_activation = np.dot(inputs,hidden_weights)
            hidden_layer_activation += hidden_bias
            hidden_layer_output = self.sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output,output_weights)
            output_layer_activation += output_bias
            predicted_output = self.sigmoid(output_layer_activation)

            #Backpropagation
            error = expected_output - predicted_output
            d_predicted_output = error * self.sigmoid_derivative(predicted_output)
            
            error_hidden_layer = d_predicted_output.dot(output_weights.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            #Updating Weights and Biases
            output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
            output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
            hidden_weights += inputs.T.dot(d_hidden_layer) * lr
            hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

        print("Final hidden weights, biases:{0},{1} ".format(*hidden_weights, *hidden_bias))
        print("Final output weights, biases:{0},{1} ".format(*output_weights, *output_bias))

        print("\nOutput from neural network after 10,000 epochs: ".format(*predicted_output))
        
if __name__ == "__main__":
    neural_network_two = XorNeuralNetork(4)
    neural_network_two.training_algorithm(1000, 0.1)