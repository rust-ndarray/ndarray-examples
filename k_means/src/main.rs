#![allow(non_snake_case)]
use ndarray::{Array, Array2, Axis, stack};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

// Import KMeans from other file ("lib.rs") in this example
use k_means::KMeans;

/// It returns a data distribution.
///
/// The data is clearly centered around two distinct points,
/// to quickly spot if something is wrong with the KMeans algorithm
/// looking at the output.
fn get_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let shape = (n_samples / 2, n_features);
    let X1: Array2<f64> = Array::random(shape, Normal::new(1000., 0.1));
    let X2: Array2<f64> = Array::random(shape, Normal::new(-1000., 0.1));
    stack(Axis(0), &[X1.view(), X2.view()]).unwrap().to_owned()
}

pub fn main() {
    let n_samples = 5000000;
    let n_features = 3;
    let n_clusters = 8;

    let X = get_data(n_samples, n_features);

    let mut k_means = KMeans::new(n_clusters);
    k_means.fit(&X);

    println!(
        "The centroids are {:.3}",
        k_means.centroids.unwrap()
    );
}
