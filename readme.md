# Neural Network Project

This project implements a neural network with flexible layer architecture. The neural network is defined by the `NeuralNetwork` struct, which contains the layer information and weights.

The `matrix.rs` file contains the implementation of the `Matrix` struct, which is used for matrix operations within the neural network.

## Neural Network Structure

The `NeuralNetwork` struct has the following fields:

```rust
struct NeuralNetwork {
    layer_info: Vec<usize>,
    weights: Vec<Matrix>,
}
```

## Usage

```rust
use crate::neural_network::NeuralNetwork;

fn main() {
    // Define the layer parameters
    let layer_info = vec![input_size, hidden_size, output_size];

    // Create a new neural network
    let mut neural_network = NeuralNetwork::new(layer_info).expect("Failed to create neural network.");

    // Perform training on the network
    let input = vec![0.5, 0.3, 0.1];
    let target = vec![0.8, 0.2];
    let learning_rate = 0.01;
    neural_network.train(input, target, learning_rate);

    // Perform a query on the trained network
    let input = vec![0.7, 0.2, 0.9];
    let output = neural_network.query(input);
    println!("Output: {:?}", output);
}
```