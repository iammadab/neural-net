use rand::distributions::Distribution;
use rand_distr::Normal;

#[derive(Debug)]
pub(crate) struct Matrix {
    row: usize,
    column: usize,
    values: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a matrix from of Vec<Vec<f64>> + validation
    /// Confirms that all inner vectors are of the same size
    /// i.e columns are the same size
    /// Returns and error if not the same
    fn new(values: Vec<Vec<f64>>) -> Result<Self, String> {
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
    fn with_value(row: usize, column: usize, value: f64) -> Self {
        let values = (0..row).map(|_| vec![value; column]).collect();
        Self {
            row,
            column,
            values,
        }
    }

    /// Given the row and column count, initializes a matrix
    /// of that size with just 0 values
    pub(crate) fn zeros(row: usize, column: usize) -> Self {
        Matrix::with_value(row, column, 0.0)
    }

    /// Generates a matrix with values sampled from a normal distribution
    /// given the mean and the standard deviation
    // TODO: potentially add support for a seed
    pub(crate) fn with_normal_distribution(row: usize, column: usize, mean: f64, std_dev: f64) -> Self {
        let normal_distribution = Normal::new(mean, std_dev).expect("Invalid parameters");
        let mut rng = rand::thread_rng();
        let zero_matrix = Self::zeros(row, column);
        zero_matrix.apply_fn(|_| normal_distribution.sample(&mut rng))
    }

    /// Given two vectors of the same size calculates the dot product
    /// Leaves vector length validation to the matrix struct
    /// hence: this method should not be made public
    fn vector_dot_product(a: &[f64], b: &[f64]) -> f64 {
        // assumes a and b are of the same size
        a.iter().zip(b.iter()).map(|(v1, v2)| v1 * v2).sum()
    }

    /// Returns the dimension of the matrix
    pub(crate) fn dim(&self) -> (usize, usize) {
        (self.row, self.column)
    }

    /// Return the nth row of a matrix as a vector
    fn get_row(&self, index: usize) -> Vec<f64> {
        self.values[index].clone()
    }

    /// Return all the row vectors
    fn get_rows(&self) -> Vec<Vec<f64>> {
        self.values.clone()
    }

    /// Return the nth column of a matrix as a vector
    fn get_column(&self, index: usize) -> Vec<f64> {
        self.values.iter().map(|row| row[index]).collect()
    }

    /// Return all the column vectors
    fn get_columns(&self) -> Vec<Vec<f64>> {
        (0..self.column).map(|col| self.get_column(col)).collect()
    }

    /// Create a new matrix that converts columns to rows
    fn transpose(&self) -> Matrix {
        Matrix::new((0..self.column).map(|i| self.get_column(i)).collect())
            .expect("values generated by us so can't error")
    }

    /// Perform matrix multiplication
    // TODO: can be made more efficient
    fn mul(&self, other: Matrix) -> Matrix {
        Matrix::new(
            self.get_rows()
                .iter()
                .map(|row| {
                    other
                        .get_columns()
                        .iter()
                        .map(|col| Matrix::vector_dot_product(row, col))
                        .collect()
                })
                .collect(),
        )
        .expect("values generated by us so can't error")
    }

    /// Apply a function to all elements in the matrix
    fn apply_fn(&self, mut fn_def: impl FnMut(&f64) -> f64) -> Matrix {
        Matrix::new(
            self.values
                .iter()
                .map(|row| row.iter().map(|val| fn_def(val)).collect())
                .collect(),
        )
        .expect("values generated by us so can't error")
    }
}

#[cfg(test)]
mod test {
    use crate::matrix::Matrix;

    #[test]
    fn build_matrix_from_values() {
        assert!(Matrix::new(vec![]).is_ok());
        assert!(Matrix::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).is_ok());
        assert!(Matrix::new(vec![vec![0.0, 0.0], vec![0.0, 0.0, 0.0]]).is_err());
    }

    #[test]
    fn zero_matrix() {
        let mat = Matrix::zeros(1, 1);
        assert_eq!(mat.values, vec![vec![0.0]]);

        let mat = Matrix::zeros(0, 0);
        assert_eq!(mat.values, Vec::<Vec<f64>>::new());

        let mat = Matrix::zeros(1, 2);
        assert_eq!(mat.values, vec![vec![0.0, 0.0]]);

        let mat = Matrix::zeros(3, 2);
        assert_eq!(
            mat.values,
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]]
        );
    }

    #[test]
    fn vector_dot_product() {
        assert_eq!(Matrix::vector_dot_product(&[1.0, 2.0], &[5.0, 7.0]), 19.0);
        assert_eq!(Matrix::vector_dot_product(&[3.0, 4.0], &[6.0, 8.0]), 50.0);
    }

    #[test]
    fn get_row() {
        let mat = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0],
        ])
        .unwrap();
        assert_eq!(mat.get_row(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(mat.get_row(1), vec![2.0, 3.0, 4.0]);
        assert_eq!(mat.get_row(2), vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn get_column() {
        let mat = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0],
        ])
        .unwrap();
        assert_eq!(mat.get_column(0), vec![1.0, 2.0, 5.0]);
        assert_eq!(mat.get_column(1), vec![2.0, 3.0, 6.0]);
        assert_eq!(mat.get_column(2), vec![3.0, 4.0, 7.0]);
    }

    #[test]
    fn transpose() {
        let mat = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0],
        ])
        .unwrap();
        let transposed_matrix = mat.transpose();
        assert_eq!(
            transposed_matrix.values,
            vec![
                vec![1.0, 2.0, 5.0],
                vec![2.0, 3.0, 6.0],
                vec![3.0, 4.0, 7.0]
            ]
        );
        assert_eq!(transposed_matrix.row, mat.column);
        assert_eq!(transposed_matrix.column, mat.row);

        let mat = Matrix::new(vec![vec![2.0, 13.0], vec![-9.0, 11.0], vec![3.0, 17.0]]).unwrap();
        let transposed_matrix = mat.transpose();
        assert_eq!(
            transposed_matrix.values,
            vec![vec![2.0, -9.0, 3.0], vec![13.0, 11.0, 17.0]]
        );
        assert_eq!(transposed_matrix.row, mat.column);
        assert_eq!(transposed_matrix.column, mat.row);
    }

    #[test]
    fn matrix_multiplication() {
        let mat_a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let mat_b = Matrix::new(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]).unwrap();
        let mat_c = mat_a.mul(mat_b);
        assert_eq!(mat_c.row, 2);
        assert_eq!(mat_c.column, 2);
        assert_eq!(mat_c.values, vec![vec![58.0, 64.0], vec![139.0, 154.0]]);

        let mat_a = Matrix::new(vec![vec![3.0, 4.0, 2.0]]).unwrap();
        let mat_b = Matrix::new(vec![
            vec![13.0, 9.0, 7.0, 15.0],
            vec![8.0, 7.0, 4.0, 6.0],
            vec![6.0, 4.0, 0.0, 3.0],
        ])
        .unwrap();
        let mat_c = mat_a.mul(mat_b);
        assert_eq!(mat_c.row, 1);
        assert_eq!(mat_c.column, 4);
        assert_eq!(mat_c.values, vec![vec![83.0, 63.0, 37.0, 75.0]]);
    }

    #[test]
    fn apply_fn() {
        let mat_a = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        assert_eq!(
            mat_a.apply_fn(|a| a * 2.0).values,
            vec![vec![2.0, 4.0, 6.0], vec![8.0, 10.0, 12.0]]
        );
    }
}
