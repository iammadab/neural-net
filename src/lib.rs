mod matrix;
use matrix::Matrix;

struct NeuralNetwork {
    layer_info: Vec<usize>,
    weights: Vec<Matrix>,
}

impl NeuralNetwork {
    /// Creates a new neural network given the layer parameters
    /// samples the weights of a node from a normal distribution
    /// with standard deviation 1/sqrt(input) and mean 0.0
    fn new(layer_info: Vec<usize>) -> Result<Self, String> {
        if layer_info.len() < 2 {
            return Err(String::from("must have both input and output layer"));
        }

        let weights = layer_info
            .windows(2)
            .map(|window| {
                let left_layer_count = window[0];
                let right_layer_count = window[1];
                let std_dev = 1.0 / (left_layer_count as f64).sqrt();
                Matrix::with_normal_distribution(right_layer_count, left_layer_count, 0.0, std_dev)
            })
            .collect::<Vec<Matrix>>();

        Ok(Self {
            layer_info,
            weights,
        })
    }

    /// Creates a neural network with given layer parameters and pre-defined weights
    /// performs validation to confirm weight dimensions are correct for layer definition
    fn new_with_weights(layer_info: Vec<usize>, weights: Vec<Matrix>) -> Result<Self, String> {
        if layer_info.len() < 2 {
            return Err(String::from("must have both input and output layer"));
        }

        if weights.len() != layer_info.len() - 1 {
            return Err(String::from("weights should be one less than the layer info"));
        }

        // we need to verify the weight dimensions
        for (window, weight) in layer_info.windows(2).zip(weights.iter()){
            // the weight dimension should be the inverse of the window
            if weight.dim() != (window[1], window[0]) {
                return Err(String::from("invalid weights for layer"));
            }
        };

        // all checks pass
        Ok(Self {
            layer_info,
            weights
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{Matrix, NeuralNetwork};

    #[test]
    fn neural_net_initialization() {
        // cannot init with less than 2 layers
        // 5 nodes for input layer but no output layer, should error
        assert!(NeuralNetwork::new(vec![5]).is_err());

        // input layer contains 2 nodes
        // just one hidden layer with 3 nodes
        // output layer contains 3 nodes
        let network = NeuralNetwork::new(vec![2, 3, 3]).unwrap();
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.weights[0].dim(), (3, 2));
        assert_eq!(network.weights[1].dim(), (3, 3));

        // build network with user defined weights
        let network = NeuralNetwork::new_with_weights(
            vec![2, 2],
            vec![Matrix::new(vec![vec![0.9, 0.3], vec![0.2, 0.3]]).unwrap()]
        ).unwrap();
    }
}

// we can have a matrix or we can have a vector
// is there a way to unify them?
// Vec or Vec
// can multiply a matrix by another matrix
// should just work
