#![allow(non_snake_case)]
use ndarray::{stack, Array, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

// Import KMeans from other file ("lib.rs") in this example
use k_means::KMeans;

/// It returns a data distribution.
///
/// The data is clearly centered around two distinct points,
/// to quickly spot if something is wrong with the KMeans algorithm
/// looking at the output.
fn get_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let shape = (n_samples / 2, n_features);
    let X1: Array2<f64> = Array::random(shape, Normal::new(1000., 0.1).unwrap());
    let X2: Array2<f64> = Array::random(shape, Normal::new(-1000., 0.1).unwrap());
    stack(Axis(0), &[X1.view(), X2.view()]).unwrap().to_owned()
}

pub fn main() {
    let n_samples = 50000;
    let n_features = 3;
    let n_clusters = 2;

    let X = get_data(n_samples, n_features);

    let mut k_means = KMeans::new(n_clusters);
    k_means.fit(&X);

    println!("The centroids are {:.3}", k_means.centroids.unwrap());
}
