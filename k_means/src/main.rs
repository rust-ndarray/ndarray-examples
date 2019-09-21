#![allow(non_snake_case)]
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_stats::DeviationExt;
use rand::distributions::{Normal, StandardNormal};

// Import KMeans from other file ("lib.rs") in this example
use k_means::KMeans;

/// It returns a data distribution.
///
fn get_data(n_samples: usize, n_features: usize) -> Array2<f64> {
    let shape = (n_samples, n_features);
    let X: Array2<f64> = Array::random(shape, StandardNormal);
    X
}

pub fn main() {
    let n_samples = 5000;
    let n_features = 3;
    let n_clusters = 2;

    let X = get_data(n_samples, n_features);

    let mut k_means = KMeans::new(n_clusters);
    k_means.fit(&X);

    println!(
        "The centroids are {:.3}",
        k_means.centroids.unwrap()
    );
}
