const Vector = class {
  //x1, x2, ..., xn
  constructor(...args) {
    this.components = args;
  }
};
Vector.add = function (vector1, vector2) {
  let new_components = [];
  for (let i = 0; i < vector1.components.length; i++) {
    new_components[i] = vector1.components[i] + vector2.components[i];
  }
  return new Vector(...new_components);
};
Vector.substract = function (vector1, vector2) {
  let new_components = [];
  for (let i = 0; i < vector1.components.length; i++) {
    new_components[i] = vector1.components[i] - vector2.components[i];
  }
  return new Vector(...new_components);
};
Vector.scale = function (vector, constant) {
  let new_components = [];
  for (let i = 0; i < vector.components.length; i++) {
    new_components[i] = vector.components[i] * constant;
  }
  return new Vector(...new_components);
};
Vector.hadamard = function (vector1, vector2) {
  let new_components = [];
  for (let i = 0; i < vector1.components.length; i++) {
    new_components[i] = vector1.components[i] * vector2.components[i];
  }
  return new Vector(...new_components);
};
Vector.apply_function = function (vector, func) {
  let new_components = [];
  for (let i = 0; i < vector.components.length; i++) {
    new_components[i] = func(vector.components[i]);
  }
  return new Vector(...new_components);
};
Vector.to_matrix = function (vector) {
  let matrix = new Matrix(vector.components);
  return Matrix.transpose(matrix);
};

const Matrix = class {
  //[x11, x12, ..., x1n], [x21, x22, ..., x2n], ..., [xm1, xm2, ..., xmn]
  //row-first: each sub-array represents a row of the matrix
  constructor(...args) {
    this.components = args;
  }
};
Matrix.substract_matrices = function (matrix1, matrix2) {
  let new_components = [];
  for (let y = 0; y < matrix1.components.length; y++) {
    new_components[y] = [];
    for (let x = 0; x < matrix1.components[0].length; x++) {
      new_components[y][x] = matrix1.components[y][x] - matrix2.components[y][x];
    }
  }
  return new Matrix(...new_components);
};
Matrix.mult_matrices = function (matrix1, matrix2) {
  let new_components = [];
  for (let y = 0; y < matrix1.components.length; y++) {
    new_components[y] = [];
  }
  for (let y = 0; y < matrix1.components.length; y++) {
    for (let x = 0; x < matrix2.components[0].length; x++) {
      let sum = 0;
      for (let i = 0; i < matrix1.components[0].length; i++) {
        sum += matrix1.components[y][i] * matrix2.components[i][x];
      }
      new_components[y][x] = sum;
    }
  }
  return new Matrix(...new_components);
};
Matrix.mult_matrix_vector = function (matrix, vector) {
  let new_components = [];
  for (let y = 0; y < matrix.components.length; y++) {
    let sum = 0;
    for (let i = 0; i < vector.components.length; i++) {
      sum += matrix.components[y][i] * vector.components[i];
    }
    new_components[y] = sum;
  }
  return new Vector(...new_components);
};
Matrix.transpose = function (matrix) {
  let new_components = [];
  for (let x = 0; x < matrix.components[0].length; x++) {
    new_components[x] = [];
  }
  for (let y = 0; y < matrix.components.length; y++) {
    for (let x = 0; x < matrix.components[0].length; x++) {
      new_components[x][y] = matrix.components[y][x];
    }
  }
  return new Matrix(...new_components);
};
Matrix.scale = function (matrix, constant) {
  let new_components = [];
  for (let y = 0; y < matrix.components.length; y++) {
    new_components[y] = [];
    for (let x = 0; x < matrix.components[0].length; x++) {
      new_components[y][x] = matrix.components[y][x] * constant;
    }
  }
  return new Matrix(...new_components);
};
Matrix.to_vector = function (matrix) {
  let transposed_matrix = Matrix.transpose(matrix);
  return new Vector(...transposed_matrix.components[0]);
};

export { Vector, Matrix };
