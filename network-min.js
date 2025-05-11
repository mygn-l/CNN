import fs from "fs";
import path from "path";
import url from "url";

import { TENSOR, QUICK_MATRIX, QUICK_VECTOR } from "../math/linear-min.js";

const Convolution_Layer = class {
  constructor(stack, height, width, kernels, activation_function, activation_derivative) {
    this.input_dimensions = [stack, height, width];

    this.kernels = kernels;

    this.padded_inputs = [];
    this.preactivations = [];
    this.activations = [];

    this.activation_function = activation_function;
    this.activation_derivative = activation_derivative;
  }
  propagate(input_tensor) {
    let activation_function = this.activation_function;

    let padding = new Array((this.kernels[0].length - 1) / 2).fill(0);
    for (let i = 0; i < input_tensor.length; i++) {
      //only odd-sized, same-sized, square kernels
      let padded_input = QUICK_MATRIX.padded_matrix(input_tensor[i], padding);
      this.padded_inputs[i] = padded_input;
    }

    for (let i = 0; i < this.kernels.length; i++) {
      let matrix_sum = QUICK_MATRIX.matrix_builder(this.input_dimensions[1], this.input_dimensions[2]);

      for (let j = 0; j < this.input_dimensions[0]; j++) {
        matrix_sum = QUICK_MATRIX.add(matrix_sum, QUICK_MATRIX.convolve(this.padded_inputs[j], this.kernels[i]));
      }

      let matrix_average = QUICK_MATRIX.scalar_mult(matrix_sum, 1 / this.input_dimensions[0]);

      this.preactivations[i] = matrix_average;

      let matrix_activated = QUICK_MATRIX.apply_map(matrix_average, function (coordinate, i, j) {
        return activation_function(coordinate);
      });

      this.activations[i] = matrix_activated;
    }

    return TENSOR.copy(this.activations);
  }
  backpropagate(derivative_cost_activation_L, learning_rate_L) {
    let activation_derivative = this.activation_derivative;

    let padding = new Array((this.kernels[0].length - 1) / 2).fill(0);

    let flipped_kernels = [];

    for (let i = 0; i < this.kernels.length; i++) {
      flipped_kernels[i] = QUICK_MATRIX.rotate180(this.kernels[i]);
    }

    let derivative_cost_preactivation_withstack_L = [];

    for (let i = 0; i < this.activations.length; i++) {
      let derivative_activation_preactivation_L_i = QUICK_MATRIX.apply_map(this.preactivations[i], function (coordinate, i, j) {
        return activation_derivative(coordinate);
      });

      let derivative_cost_preactivation_L_i = QUICK_MATRIX.hadamard(derivative_cost_activation_L[i], derivative_activation_preactivation_L_i);

      derivative_cost_preactivation_withstack_L[i] = QUICK_MATRIX.scalar_mult(derivative_cost_preactivation_L_i, 1 / this.input_dimensions[0]);
    }

    let padded_inputs_sum = QUICK_MATRIX.matrix_builder(this.padded_inputs[0].length, this.padded_inputs[0][0].length);

    for (let i = 0; i < this.input_dimensions[0]; i++) {
      padded_inputs_sum = QUICK_MATRIX.add(padded_inputs_sum, this.padded_inputs[i]);
    }

    for (let i = 0; i < this.kernels.length; i++) {
      let derivative_cost_weight_L_i = TENSOR.apply_map(this.kernels[i], function (coordinate, address) {
        let derivative_preactivation_weight_L_i_ab = QUICK_MATRIX.submatrix(
          padded_inputs_sum,
          [address[0], address[0] + derivative_cost_preactivation_withstack_L[i].length],
          [address[1], address[1] + derivative_cost_preactivation_withstack_L[i][0].length]
        );

        let derivative_cost_weight_L_i_ab = QUICK_MATRIX.internal_sum(
          TENSOR.hadamard(derivative_cost_preactivation_withstack_L[i], derivative_preactivation_weight_L_i_ab)
        );

        return derivative_cost_weight_L_i_ab;
      });

      let dkernel_L_i = TENSOR.scalar_mult(derivative_cost_weight_L_i, -learning_rate_L);

      this.kernels[i] = TENSOR.add(this.kernels[i], dkernel_L_i);
    }

    let derivative_cost_activation_Lminus1 = [];

    for (let i = 0; i < this.input_dimensions[0]; i++) {
      derivative_cost_activation_Lminus1[i] = QUICK_MATRIX.convolve(
        QUICK_MATRIX.padded_matrix(derivative_cost_preactivation_withstack_L[i], padding),
        flipped_kernels[i]
      );
    }

    return derivative_cost_activation_Lminus1;
  }
};

const Average_Pooling_Layer = class {
  constructor(kernel, stride) {
    this.kernel = kernel;
    this.stride = {
      vertical: stride,
      horizontal: stride,
    };
  }
  propagate(input_tensor) {
    //average pool, I will change to max pool later
    let output_tensor = [];
    for (let i = 0; i < input_tensor.length; i++) {
      output_tensor[i] = QUICK_MATRIX.convolve(input_tensor[i], this.kernel, this.stride);
    }

    return output_tensor;
  }
  backpropagate(derivative_cost_activation_L, learning_rate_L) {
    let stride = this.stride;
    let kernel = this.kernel;

    //assuming we have a kernel that averages values
    let recreated_tensor = TENSOR.tensor_builder(
      [
        derivative_cost_activation_L.length,
        derivative_cost_activation_L[0].length * stride.vertical,
        derivative_cost_activation_L[0][0].length * stride.horizontal,
      ],
      function (address) {
        return (
          derivative_cost_activation_L[address[0]][Math.floor(address[1] / stride.vertical)][Math.floor(address[2] / stride.horizontal)] *
          kernel[address[1] % stride.vertical][address[2] % stride.horizontal]
        );
      }
    );

    return recreated_tensor;
  }
};

const Max_Pooling_Layer = class {
  constructor(stride) {
    this.stride = {
      vertical: stride,
      horizontal: stride,
    };

    this.maximums;
  }
  propagate(input_tensor) {
    this.maximums = TENSOR.tensor_builder([input_tensor.length, input_tensor[0].length, input_tensor[0][0].length]);
    let maximums = this.maximums;

    let stride = this.stride;

    let output_tensor = [];
    for (let k = 0; k < input_tensor.length; k++) {
      output_tensor[k] = QUICK_MATRIX.matrix_builder(
        input_tensor[k].length / stride.vertical,
        input_tensor[k][0].length / stride.horizontal,
        function (i, j) {
          let sub_matrix = QUICK_MATRIX.submatrix(
            input_tensor[k],
            [i * stride.vertical, (i + 1) * stride.vertical],
            [j * stride.horizontal, (j + 1) * stride.horizontal]
          );
          let max = sub_matrix[0][0];
          let max_address = [0, 0];
          TENSOR.traverse_tensor(sub_matrix, function (coordinate, address) {
            if (coordinate > max) {
              max = coordinate;
              max_address = address;
            }
          });

          maximums[k][i * stride.vertical + max_address[0]][j * stride.horizontal + max_address[1]] = 1;

          return max;
        }
      );
    }

    return output_tensor;
  }
  backpropagate(derivative_cost_activation_L, learning_rate_L) {
    let stride = this.stride;
    let maximums = this.maximums;

    //assuming we have a kernel that averages values
    let recreated_tensor = TENSOR.tensor_builder(
      [
        derivative_cost_activation_L.length,
        derivative_cost_activation_L[0].length * stride.vertical,
        derivative_cost_activation_L[0][0].length * stride.horizontal,
      ],
      function (address) {
        if (TENSOR.coordinate_from_address(maximums, address) === 1) {
          return derivative_cost_activation_L[address[0]][Math.floor(address[1] / stride.vertical)][Math.floor(address[2] / stride.horizontal)];
        } else {
          return 0;
        }
      }
    );

    return recreated_tensor;
  }
};

const Flattening_Layer = class {
  constructor(stack, height, width) {
    this.input_dimensions = [stack, height, width];
  }
  propagate(input_tensor) {
    let flattened = [];
    for (let matrix of input_tensor) {
      for (let row of matrix) {
        flattened = flattened.concat(row);
      }
    }

    return flattened;
  }
  backpropagate(derivative_cost_activation_L, learning_rate_L) {
    let input_dimensions = this.input_dimensions;

    return TENSOR.tensor_builder(input_dimensions, function (address) {
      return derivative_cost_activation_L[address[0] * input_dimensions[1] * input_dimensions[2] + address[1] * input_dimensions[2] + address[2]];
    });
  }
};

const Full_Layer = class {
  constructor(input_size, output_size, activation_function, activation_derivative, initial_range) {
    this.bias = TENSOR.tensor_builder([output_size], function (address) {
      return Math.random() * initial_range;
    });
    this.weights = QUICK_MATRIX.matrix_builder(output_size, input_size, function (i, j) {
      return Math.random() * initial_range;
    });

    this.activation_function = activation_function;
    this.activation_derivative = activation_derivative;

    this.inputs;
    this.preactivations;
    this.activations;
  }
  propagate(input_vector) {
    this.inputs = TENSOR.copy(input_vector);

    this.preactivations = QUICK_VECTOR.add(QUICK_MATRIX.matrix_mult_vector(this.weights, input_vector), this.bias);

    let activation_function = this.activation_function;

    this.activations = QUICK_VECTOR.apply_map(this.preactivations, function (coordinate, i) {
      return activation_function(coordinate);
    });

    return TENSOR.copy(this.activations);
  }
  backpropagate(derivative_cost_activation_L, learning_rate_L) {
    let activation_derivative = this.activation_derivative;

    let derivative_activation_preactivation_L = QUICK_VECTOR.apply_map(this.preactivations, function (coordinate, i) {
      return activation_derivative(coordinate);
    });

    let derivative_cost_preactivation_L = QUICK_VECTOR.hadamard(derivative_cost_activation_L, derivative_activation_preactivation_L);

    let derivative_cost_weights_L = TENSOR.product(derivative_cost_preactivation_L, this.inputs);

    let derivative_cost_activation_Lminus1 = QUICK_MATRIX.covector_mult_matrix(derivative_cost_preactivation_L, this.weights); //multiply the weights before they change

    let dweights_L = QUICK_MATRIX.scalar_mult(derivative_cost_weights_L, -learning_rate_L);
    this.weights = QUICK_MATRIX.add(this.weights, dweights_L);

    let dbias_L = QUICK_VECTOR.scalar_mult(derivative_cost_preactivation_L, -learning_rate_L);
    this.bias = QUICK_VECTOR.add(this.bias, dbias_L);

    return derivative_cost_activation_Lminus1;
  }
};

const Convolutional_Network = class {
  constructor(...interlayers) {
    this.interlayers = interlayers;

    this.final_activations;
  }
  propagate(input_tensor) {
    let current_tensor = TENSOR.copy(input_tensor);

    //console.table(current_tensor[0]);

    for (let interlayer of this.interlayers) {
      current_tensor = interlayer.propagate(current_tensor);

      /*
      if (interlayer instanceof Convolution_Layer || interlayer instanceof Max_Pooling_Layer) {
        console.table(current_tensor[0]);
      } else {
        console.table(current_tensor);
      }
      */
    }

    this.final_activations = TENSOR.copy(current_tensor);

    return current_tensor;
  }
  backpropagate(expected_output, learning_rates) {
    let derivative_cost_activation_L = QUICK_VECTOR.subtract(this.final_activations, expected_output);

    for (let L = this.interlayers.length - 1; L >= 0; L--) {
      let interlayer = this.interlayers[L];

      /*
      if (interlayer instanceof Convolution_Layer || interlayer instanceof Max_Pooling_Layer) {
        console.table(derivative_cost_activation_L[0]);
      } else {
        console.table(derivative_cost_activation_L);
      }
        */

      derivative_cost_activation_L = interlayer.backpropagate(derivative_cost_activation_L, learning_rates[L]);
    }

    return derivative_cost_activation_L;
  }
  store_weights() {
    const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const store_path = path.join(__dirname, "weights.json");

    const weights = [];
    for (let L = 0; L < this.interlayers.length; L++) {
      if (this.interlayers[L] instanceof Convolution_Layer) {
        weights[L] = {
          kernels: this.interlayers[L].kernels,
        };
      } else if (this.interlayers[L] instanceof Full_Layer) {
        weights[L] = {
          weights: this.interlayers[L].weights,
          bias: this.interlayers[L].bias,
        };
      } else {
        weights[L] = {};
      }
    }

    fs.writeFileSync(store_path, JSON.stringify(weights));
  }
  read_weights() {
    const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const store_path = path.join(__dirname, "weights.json");

    const weights = JSON.parse(fs.readFileSync(store_path, "utf-8"));

    for (let L = 0; L < this.interlayers.length; L++) {
      if (this.interlayers[L] instanceof Convolution_Layer) {
        this.interlayers[L].kernels = weights[L].kernels;
      } else if (this.interlayers[L] instanceof Full_Layer) {
        this.interlayers[L].bias = weights[L].bias;
        this.interlayers[L].weights = weights[L].weights;
      }
    }
  }
  read_csv(from, to) {
    const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const store_path = path.join(__dirname, "mnist_train.csv");

    const text = fs.readFileSync(store_path, "utf-8");

    let arrays = [];
    let expected_digits = [];
    let array = []; //flattened
    let word = "";
    for (let i = 0; i < text.length; i++) {
      if (text[i] === "\n") {
        let expected_digit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        expected_digit[array[0]] = 1;
        expected_digits.push(expected_digit);
        array[0] = 0;

        arrays.push(array);

        array = [];
      } else if (text[i] === ",") {
        array.push(parseInt(word));
        word = "";
      } else {
        word = word.concat(text[i]);
      }
    }

    expected_digits = expected_digits.slice(from, to);
    arrays = arrays.slice(from, to);

    let flatten_layer = new Flattening_Layer(1, 28, 28);
    let normalized_tensors = [];

    for (let flattened_input_tensor of arrays) {
      let unflattened_matrix = flatten_layer.backpropagate(flattened_input_tensor, 1)[0];
      let normalized_tensor = [QUICK_MATRIX.scalar_mult(unflattened_matrix, 1 / 256)];
      normalized_tensors.push(normalized_tensor);
    }

    return {
      expected_digits: expected_digits,
      normalized_tensors: normalized_tensors,
    };
  }
};

export { Convolutional_Network, Convolution_Layer, Max_Pooling_Layer, Average_Pooling_Layer, Flattening_Layer, Full_Layer };
