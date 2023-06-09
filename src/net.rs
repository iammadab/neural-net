use crate::matrix::Matrix;

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
            return Err(String::from(
                "weights should be one less than the layer info",
            ));
        }

        // we need to verify the weight dimensions
        for (window, weight) in layer_info.windows(2).zip(weights.iter()) {
            // the weight dimension should be the inverse of the window
            if weight.dim() != (window[1], window[0]) {
                return Err(String::from("invalid weights for layer"));
            }
        }

        // all checks pass
        Ok(Self {
            layer_info,
            weights,
        })
    }

    /// Performs a feed forward but only returns the final output
    fn query(&self, input: Vec<f64>) -> Vec<f64> {
        self.feed_forward(input)
            .last()
            .expect("must have output")
            .transpose()
            .get_row(0)
    }

    /// Runs the network on some input
    /// Returns the output from each layer (input layer inclusive)
    fn feed_forward(&self, input: Vec<f64>) -> Vec<Matrix> {
        let input_vector = Matrix::row_matrix_from_vector(input);
        let mut result = vec![input_vector];
        for weight in &self.weights {
            result.push(
                weight
                    .mul(result.last().expect("not empty"))
                    .apply_fn(NeuralNetwork::sigmoid),
            )
        }
        result
    }

    /// Given a training sample, updates the weight of the network to minimize the error
    fn train(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) {
        let weight_deltas = self.get_weight_deltas(input, target, learning_rate);
        let updated_weights = self
            .weights
            .iter()
            .zip(weight_deltas)
            .map(|(w, wd)| w.clone() + wd.clone())
            .collect::<Vec<Matrix>>();
        self.weights = updated_weights;
    }

    /// Performs gradient descent to figure out how to update the weights of the network
    /// Returns matrix update values for each weight matrix
    fn get_weight_deltas(
        &self,
        input: Vec<f64>,
        target: Vec<f64>,
        learning_rate: f64,
    ) -> Vec<Matrix> {
        // perform feed forward to get the output
        let outputs = self.feed_forward(input);
        let targets = Matrix::row_matrix_from_vector(target);

        // compare the output to the target list to figure out the error
        let output_errors = targets - outputs.last().unwrap().clone();

        // back-propagate to figure out the error for each layer
        let errors = self.back_propagate_errors(output_errors);

        // for gradient descent, we need another matrix that subtract all
        // the output values from 1
        let one_minus_outputs = outputs
            .iter()
            .map(|out| out.apply_fn(|v| 1.0 - v))
            .collect::<Vec<Matrix>>();

        let mut weight_updates = vec![];

        // update each weight connection matrix
        for i in (1..outputs.len()).rev() {
            let layer_error = errors.get(i - 1).unwrap().clone();
            let layer_output = outputs.get(i).unwrap().clone();
            let one_minus_layer_output = one_minus_outputs.get(i).unwrap().clone();

            let a = layer_error * layer_output * one_minus_layer_output;
            let b = outputs[i - 1].clone().transpose();

            let delta = a.mul(&b).apply_fn(|f| f * learning_rate);

            // we insert to the front of the list because the updates
            // were made from the back (rev)
            weight_updates.insert(0, delta)
        }

        weight_updates
    }

    /// Returns the error for each layer expect the input layer
    fn back_propagate_errors(&self, output_error: Matrix) -> Vec<Matrix> {
        let mut errors = vec![output_error];
        // we skip the first weight as we don't care about errors for the input node
        // we reverse the iterator because we are starting from the output layer
        for weight in self.weights.iter().skip(1).rev() {
            let error = weight.transpose().mul(&errors[0]);
            errors.insert(0, error);
        }
        errors
    }

    /// Sigmoid function
    fn sigmoid(input: &f64) -> f64 {
        1.0 / (1.0 + f64::exp(-input))
    }
}

#[cfg(test)]
mod test {
    use crate::{matrix::Matrix, net::NeuralNetwork};

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
            vec![Matrix::new(vec![vec![0.9, 0.4], vec![0.2, 0.3]]).unwrap()],
        )
        .unwrap();
    }

    #[test]
    fn sigmoid() {
        assert_eq!(NeuralNetwork::sigmoid(&0.0), 0.5);
        assert_eq!(NeuralNetwork::sigmoid(&3.0), 0.9525741268224334);
    }

    #[test]
    fn query() {
        // Two layers
        let network = NeuralNetwork::new_with_weights(
            vec![2, 2],
            vec![Matrix::new(vec![vec![0.9, 0.3], vec![0.2, 0.8]]).unwrap()],
        )
        .unwrap();
        let output = network.query(vec![1.0, 0.5]);
        assert_eq!(output, vec![0.740774899182154, 0.6456563062257954]);

        // Three layer
        let network = NeuralNetwork::new_with_weights(
            vec![3, 3, 3],
            vec![
                Matrix::new(vec![
                    vec![0.9, 0.3, 0.4],
                    vec![0.2, 0.8, 0.2],
                    vec![0.1, 0.5, 0.6],
                ])
                .unwrap(),
                Matrix::new(vec![
                    vec![0.3, 0.7, 0.5],
                    vec![0.6, 0.5, 0.2],
                    vec![0.8, 0.1, 0.9],
                ])
                .unwrap(),
            ],
        )
        .unwrap();
        let output = network.query(vec![0.9, 0.1, 0.8]);
        assert_eq!(
            output,
            vec![0.7263033450139793, 0.7085980724248232, 0.778097059561142]
        );
    }

    #[test]
    fn back_propagate_errors() {
        let network = NeuralNetwork::new_with_weights(
            vec![2, 2, 2],
            vec![
                Matrix::new(vec![vec![3.0, 2.0], vec![1.0, 7.0]]).unwrap(),
                Matrix::new(vec![vec![2.0, 3.0], vec![1.0, 4.0]]).unwrap(),
            ],
        )
        .unwrap();

        let errors = network.back_propagate_errors(Matrix::row_matrix_from_vector(vec![0.8, 0.5]));
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].values(), vec![vec![2.1], vec![4.4]]);
        assert_eq!(errors[1].values(), vec![vec![0.8], vec![0.5]]);
    }

    #[test]
    fn get_weight_deltas() {
        let mut network = NeuralNetwork::new_with_weights(
            vec![3, 3, 3],
            vec![
                Matrix::new(vec![
                    vec![-0.21787096, 0.09627831, 0.39421614],
                    vec![1.10930855, -0.11375925, -0.43871621],
                    vec![-1.20612302, -0.02099995, -0.70306782],
                ])
                .unwrap(),
                Matrix::new(vec![
                    vec![-0.82497876, -0.09845416, 0.67026903],
                    vec![-0.61264944, -0.81319795, 0.48471442],
                    vec![-0.05751839, -1.00243301, -0.17735394],
                ])
                .unwrap(),
            ],
        )
        .unwrap();
        let output = network.query(vec![1.0, 0.5, -1.5]);

        let weight_deltas =
            network.get_weight_deltas(vec![1.0, 0.5, 0.1], vec![0.9, 0.8, 0.7], 0.3);

        network.train(vec![1.0, 0.5, 0.1], vec![0.9, 0.8, 0.7], 0.3);

        let output = network.query(vec![1.0, 0.5, -1.5]);
        assert_eq!(
            output,
            vec![0.5019062063805478, 0.35040324838192555, 0.28482826169784886]
        );
    }
}
