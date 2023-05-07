mod matrix;
use matrix::Matrix;

struct NeuralNetwork {
    layer_info: Vec<usize>,
    weights: Vec<Matrix>,
}

impl NeuralNetwork {
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
}

#[cfg(test)]
mod test {
    use crate::NeuralNetwork;

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
    }
}

// we can have a matrix or we can have a vector
// is there a way to unify them?
// Vec or Vec
// can multiply a matrix by another matrix
// should just work
