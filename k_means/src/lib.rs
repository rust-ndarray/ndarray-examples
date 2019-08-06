#![allow(non_snake_case)]
use ndarray::{Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use rand::distributions::{Distribution, Uniform};

pub struct KMeans {
    pub centroids: Option<Array2<f64>>,
    n_clusters: u16,
}

impl KMeans {
    pub fn new(n_clusters: u16) -> KMeans {
        KMeans {
            centroids: None,
            n_clusters,
        }
    }

    /// Given an input matrix `X`, with shape `(n_samples, n_features)`
    /// `fit` determines a set of centroids based on the training data distribution.
    ///
    /// `self` is modified in place, nothing is returned.
    pub fn fit<A, B>(&mut self, X: &ArrayBase<A, Ix2>)
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();
        assert!(
            n_samples >= self.n_clusters as usize,
            "We need more sample points than clusters!"
        );

        // Initialisation
        let centroids = KMeans::get_random_centroids(self.n_clusters, X);
        let cluster_memberships = X.map_axis(Axis(0), |sample| {
            KMeans::find_closest_centroid(&centroids, &sample)
        });
    }

    fn get_random_centroids<A>(n_clusters: u16, X: &ArrayBase<A, Ix2>) -> Array2<f64>
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();
        let distribution = Uniform::from(0..n_samples);
        let mut rng = rand::thread_rng();
        let indices: Vec<_> = (0..n_clusters)
            .map(|_| distribution.sample(&mut rng))
            .collect();
        X.select(Axis(0), &indices)
    }

    fn find_closest_centroid<A, B>(
        centroids: &ArrayBase<A, Ix2>,
        sample: &ArrayBase<B, Ix1>,
    ) -> usize
    where
        A: Data<Elem = f64>,
        B: Data<Elem = f64>,
    {
        let mut iterator = centroids.genrows().into_iter();

        let first_centroid = iterator.next().expect("No centroids - degenerate case!");
        let mut closest_index = 0;
        let mut minimum_distance = sample.sq_l2_dist(&first_centroid).unwrap();

        for (index, centroid) in iterator.enumerate() {
            let distance = sample.sq_l2_dist(&centroid).unwrap();
            if distance < minimum_distance {
                minimum_distance = distance;
                closest_index = index;
            }
        }

        closest_index
    }
}
