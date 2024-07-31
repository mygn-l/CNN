import fs from "fs";

import { Vector, Matrix } from "./lin-alg.js";

const Neural_network = class {
  constructor(activation_function, layer_thickness = [1, 1]) {
    this.biases = []; //array of vectors, first element is bias of first hidden layer
    this.weights = []; //array of matrices, first element is weights from input to hidden layer

    for (let i = 0; i < layer_thickness.length - 1; i++) {
      let vector_zeros = [];
      for (let j = 0; j < layer_thickness[i + 1]; j++) {
        vector_zeros[j] = Math.random();
      }
      this.biases[i] = new Vector(...vector_zeros);

      let matrix_zeros = [];
      for (let y = 0; y < layer_thickness[i + 1]; y++) {
        matrix_zeros[y] = [];
        for (let x = 0; x < layer_thickness[i]; x++) {
          matrix_zeros[y][x] = Math.random();
        }
      }
      this.weights[i] = new Matrix(...matrix_zeros);
    }

    this.activation_function = activation_function;

    this.number_of_layers = layer_thickness.length;

    this.current_activations = [];
  }
  run(inputs) {
    let current_activations = this.current_activations;
    let number_of_layers = this.number_of_layers;
    let biases = this.biases;
    let weights = this.weights;
    let activation_function = this.activation_function;

    let propagate = function (current_layer, index) {
      current_activations[index] = current_layer;

      if (index < number_of_layers - 1) {
        let next_layer = Vector.apply_function(
          Vector.add(biases[index], Matrix.mult_matrix_vector(weights[index], current_layer)),
          activation_function
        );

        propagate(next_layer, index + 1);
      }
    };

    propagate(inputs, 0);

    return current_activations[current_activations.length - 1];
  }
  train(inputs, desired_outputs, learning_rate) {
    let final_activations = this.run(inputs);
    let initial_cost_to_activation_derivatives = Vector.substract(
      final_activations,
      desired_outputs
    );

    let current_activations = this.current_activations;
    let number_of_layers = this.number_of_layers;
    let biases = this.biases;
    let weights = this.weights;

    let backpropagate = function (current_cost_to_activation_derivatives, index_from_end) {
      let activation_function_derivatives = Vector.apply_function(
        current_activations[current_activations.length - 1 - index_from_end],
        Neural_network.sigmoid_derivative_according_activation
      );

      let cost_to_preactivation_derivatives = Vector.hadamard(
        current_cost_to_activation_derivatives,
        activation_function_derivatives
      );

      let preactivation_to_weights_derivatives =
        current_activations[current_activations.length - 2 - index_from_end];

      let cost_to_weight_derivatives = Matrix.mult_matrices(
        Vector.to_matrix(cost_to_preactivation_derivatives),
        Matrix.transpose(Vector.to_matrix(preactivation_to_weights_derivatives))
      );

      weights[weights.length - 1 - index_from_end] = Matrix.scale(
        Matrix.substract_matrices(
          weights[weights.length - 1 - index_from_end],
          cost_to_weight_derivatives
        ),
        learning_rate
      );

      let cost_to_bias_derivatives = cost_to_preactivation_derivatives;

      biases[biases.length - 1 - index_from_end] = Vector.scale(
        Vector.substract(biases[biases.length - 1 - index_from_end], cost_to_bias_derivatives),
        learning_rate
      );

      if (index_from_end < number_of_layers - 2) {
        let preactivation_to_previous_activation_derivatives = Matrix.transpose(
          weights[weights.length - 1 - index_from_end]
        );

        let next_cost_to_activation_derivatives = Matrix.mult_matrix_vector(
          preactivation_to_previous_activation_derivatives,
          cost_to_preactivation_derivatives
        );

        backpropagate(next_cost_to_activation_derivatives, index_from_end + 1);
      }
    };

    backpropagate(initial_cost_to_activation_derivatives, 0);
  }
  read_file(path_to_file) {
    try {
      let file = fs.readFileSync(path_to_file);
      let parsed = JSON.parse(file);
      this.weights = parsed.weights;
      this.biases = parsed.biases;
    } catch {}
  }
  write_file(path_to_file) {
    let file_to_store = {
      weights: this.weights,
      biases: this.biases,
    };
    let stringed = JSON.stringify(file_to_store);
    fs.writeFileSync(path_to_file, stringed);
  }
};

Neural_network.sigmoid = function (input) {
  return 1 / (1 + Math.exp(-input));
};
Neural_network.sigmoid_derivative_according_activation = function (activation) {
  return activation * (1 - activation);
};

export default Neural_network;
