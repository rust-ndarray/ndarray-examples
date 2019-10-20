#![allow(non_snake_case)]
use ndarray::{stack, Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;


use svm::SupportVectorMachine;

/// It returns a data distribution.
///
/// The data is clearly centered around two distinct points,
/// to quickly spot if something is wrong with the KMeans algorithm
/// looking at the output.
fn get_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Vec<bool>) {
    let shape = (n_samples / 2, n_features);
    let X1: Array2<f64> = Array::random(shape, Normal::new(100., 5.0).unwrap());
    let X2: Array2<f64> = Array::random(shape, Normal::new(-100., 5.0).unwrap());

    (
        stack(Axis(0), &[X1.view(), X2.view()]).unwrap().to_owned(),
        (0..n_samples).map(|i| i < n_samples/2).collect()
    )
}

fn main() {
    let (X,y) = get_data(1000, 100);

    let mut svm = SupportVectorMachine::new();
    svm.fit(&X, &y);

    println!("{:?}", svm.predict(&X));
}
