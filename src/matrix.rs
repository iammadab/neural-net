struct Matrix {
    row: usize,
    column: usize,
    values: Vec<Vec<u8>>,
}

impl Matrix {
    /// Creates a matrix from of Vec<Vec<u8>> + validation
    /// Confirms that all inner vectors are of the same size
    /// i.e columns are the same size
    /// Returns and error if not the same
    fn new(values: Vec<Vec<u8>>) -> Result<Self, String> {
        let column = values.get(0).map(|a| a.len()).unwrap_or(0);

        let columns_not_equal = values.iter().any(|a| a.len() != column);
        if columns_not_equal {
            return Err(String::from(
                "invalid data: columns should be of the same size",
            ));
        }

        Ok(Self {
            row: values.len(),
            column,
            values,
        })
    }

    /// Given the row and column count, initializes a matrix
    /// of that size and sets the elements to the given value
    fn with_value(row: usize, column: usize, value: u8) -> Self {
        let values = (0..row).map(|_| vec![value; column]).collect();
        Self {
            row,
            column,
            values,
        }
    }

    /// Given the row and column count, initializes a matrix
    /// of that size with just 0 values
    fn zeros(row: usize, column: usize) -> Self {
        Matrix::with_value(row, column, 0)
    }

    /// Given two vectors of the same size calculates the dot product
    /// Leaves vector length validation to the matrix struct
    /// hence: this method should not be made public
    fn vector_dot_product(a: Vec<u8>, b: Vec<u8>) -> u8 {
        // assumes a and b are of the same size
        a.iter().zip(b.iter()).map(|(v1, v2)| v1 * v2).sum()
    }

    /// Return the nth row of a matrix as a vector
    fn get_row(&self, index: usize) -> Vec<u8> {
        self.values[index].clone()
    }

    fn get_column(&self, index: usize) -> Vec<u8> {
        self.values.iter().map(|row| row[index]).collect()
    }
}

#[cfg(test)]
mod test {
    use crate::matrix::Matrix;

    #[test]
    fn build_matrix_from_values() {
        assert!(Matrix::new(vec![]).is_ok());
        assert!(Matrix::new(vec![vec![0, 0], vec![0, 0]]).is_ok());
        assert!(Matrix::new(vec![vec![0, 0], vec![0, 0, 0]]).is_err());
    }

    #[test]
    fn zero_matrix() {
        let mat = Matrix::zeros(1, 1);
        assert_eq!(mat.values, vec![vec![0]]);

        let mat = Matrix::zeros(0, 0);
        assert_eq!(mat.values, Vec::<Vec<u8>>::new());

        let mat = Matrix::zeros(1, 2);
        assert_eq!(mat.values, vec![vec![0, 0]]);

        let mat = Matrix::zeros(3, 2);
        assert_eq!(mat.values, vec![vec![0, 0], vec![0, 0], vec![0, 0]]);
    }

    #[test]
    fn vector_dot_product() {
        assert_eq!(Matrix::vector_dot_product(vec![1, 2], vec![5, 7]), 19);
        assert_eq!(Matrix::vector_dot_product(vec![3, 4], vec![6, 8]), 50);
    }

    #[test]
    fn get_row() {
        let mat = Matrix::new(vec![vec![1, 2, 3], vec![2, 3, 4], vec![5, 6, 7]]).unwrap();
        assert_eq!(mat.get_row(0), vec![1, 2, 3]);
        assert_eq!(mat.get_row(1), vec![2, 3, 4]);
        assert_eq!(mat.get_row(2), vec![5, 6, 7]);
    }

    #[test]
    fn get_column() {
        let mat = Matrix::new(vec![vec![1, 2, 3], vec![2, 3, 4], vec![5, 6, 7]]).unwrap();
        assert_eq!(mat.get_column(0), vec![1, 2, 5]);
        assert_eq!(mat.get_column(1), vec![2, 3, 6]);
        assert_eq!(mat.get_column(2), vec![3, 4, 7]);
    }
}
