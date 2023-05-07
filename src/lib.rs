mod matrix;

struct NeuralNetwork {
    // we need the number of input nodes, number of hidden nodes and number of output node
    // weights: [(u8, u8)]
}

impl NeuralNetwork {
    fn new(node_counts: &[u8]) -> Result<Self, String> {
        if node_counts.len() < 2 {
            return Err(String::from("must have both input and output layer"));
        }

        let weights = node_counts
            .windows(2)
            .map(|window| (window[1], window[0]))
            .collect::<Vec<(u8, u8)>>();
        dbg!(weights);

        // we need to take the node counts in pairs of two
        // add code here
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::NeuralNetwork;

    #[test]
    fn test_neural_net_initialization() {
        // cannot init with less than 2 layers
        // 5 nodes for input layer but no output layer, should error
        assert!(NeuralNetwork::new(&[5]).is_err());

        // input layer contains 3 nodes
        // just one hidden layer with 3 nodes
        // output layer contains 3 nodes
        let network = NeuralNetwork::new(&[3, 3, 3]).unwrap();
    }
}

// we can have a matrix or we can have a vector
// is there a way to unify them?
// Vec or Vec
// can multiply a matrix by another matrix
// should just work
