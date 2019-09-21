#![allow(non_snake_case)]
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use std::collections::HashMap;

/// K-means clustering aims to partition a set of observations
/// into `self.n_clusters` clusters, where each observation belongs
/// to the cluster with the nearest mean.
///
/// The mean of the points within a cluster is called *centroid*.
///
/// Given the set of `centroids`, you can assign an observation to a cluster
/// choosing the nearest centroid.
///
/// Details on the algorithm can be found [here](https://en.wikipedia.org/wiki/K-means_clustering).
///
/// We are implementing the _standard algorithm_.
pub struct KMeans {
    /// Our set of centroids.
    ///
    /// Before `fit` is called, it's set to `None`.
    ///
    /// Once `fit` is called, we will have our set of centroids: the `centroids` matrix
    /// has shape `(n_clusters, n_features)`.
    pub centroids: Option<Array2<f64>>,
    /// The number of clusters we are trying to subdivide our observations into.
    /// It's set before-hand.
    n_clusters: u16,
}

impl KMeans {
    /// The number of clusters we are looking for has to be chosen before-hand,
    /// hence we take it as a constructor parameter of our `KMeans` struct.
    pub fn new(n_clusters: u16) -> KMeans {
        KMeans {
            centroids: None,
            n_clusters,
        }
    }

    /// Given an input matrix `X`, with shape `(n_samples, n_features)`
    /// `fit` determines `self.n_clusters` centroids based on the training data distribution.
    ///
    /// `self` is modified in place (`self.centroids` is mutated), nothing is returned.
    pub fn fit<A>(&mut self, X: &ArrayBase<A, Ix2>)
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();
        assert!(
            n_samples >= self.n_clusters as usize,
            "We need more sample points than clusters!"
        );

        let tolerance = 1e-3;

        // Initialisation: we use the Forgy method - we pick `self.n_clusters` random
        // observations and we use them as our first set of centroids.
        let mut centroids = KMeans::get_random_centroids(self.n_clusters, X);

        // Keep repeating the assignment-update steps until we have convergence
        loop {
            // Assignment step: associate each sample to the closest centroid
            let cluster_memberships = X.map_axis(Axis(1), |sample| {
                KMeans::find_closest_centroid(&centroids, &sample)
            });

            // Update step: using the newly computed `cluster_memberships`,
            // compute the new centroids, the means of our clusters
            let new_centroids =
                KMeans::compute_centroids(&X, &cluster_memberships);

            // Check the convergence condition: if the new centroids,
            // after the assignment-update cycle, are closer to the old centroids
            // than a pre-established tolerance we are finished.
            let distance = centroids.sq_l2_dist(&new_centroids).unwrap();
            let has_converged = distance < tolerance;

            centroids = new_centroids;

            if has_converged {
                break;
            }
        }

        // Set `self.centroids` to the outcome of our optimisation process.
        self.centroids = Some(centroids);
    }

    /// Given our observations, `X`, and the index to which each observation belongs,
    /// stored in `cluster_memberships`,
    /// we want to compute the mean of all observations in each cluster.
    fn compute_centroids<A, B>(
        X: &ArrayBase<A, Ix2>,
        cluster_memberships: &ArrayBase<B, Ix1>,
    ) -> Array2<f64>
    where
        A: Data<Elem = f64>,
        B: Data<Elem = usize>,
    {
        let (_, n_features) = X.dim();
        // `centroids` is a cluster index -> rolling mean mapping.
        // We will update it while we iterate over our observations.
        let mut centroids: HashMap<usize, RollingMean> = HashMap::new();

        // We iterate over our observations and cluster memberships in lock-step.
        let iterator = X.genrows().into_iter().zip(cluster_memberships.iter());
        for (row, cluster_index) in iterator {
            // If we have already encountered an observation that belongs to the
            // `cluster_index`th cluster, we retrieve the current rolling mean (our new centroid)
            // and we update it using the current observation.
            if let Some(rolling_mean) = centroids.get_mut(cluster_index) {
                rolling_mean.accumulate(&row);
            } else {
                // If we have not yet encountered an observation that belongs to the
                // `cluster_index`th cluster, we set its centroid to `row`,
                // initialising our `RollingMean` accumulator.
                let new_centroid = RollingMean::new(row.to_owned());
                // .to_owned takes our `row` view as input and returns an owned array.
                centroids.insert(*cluster_index, new_centroid);
            }
        }

        // Convert our `HashMap` into a 2d array.
        let mut new_centroids: Array2<f64> = Array2::zeros((centroids.len(), n_features));
        for (cluster_index, centroid) in centroids.into_iter() {
            let mut new_centroid = new_centroids.index_axis_mut(Axis(0), cluster_index);
            // .assign sets each element of `new_centroid`
            // to the corresponding element in `centroid.current_mean`.
            new_centroid.assign(&centroid.current_mean);
        }

        new_centroids
    }

    fn get_random_centroids<A>(n_clusters: u16, X: &ArrayBase<A, Ix2>) -> Array2<f64>
    where
        A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();
        let mut rng = rand::thread_rng();
        let indices = rand::seq::index::sample(&mut rng, n_samples, n_clusters as usize).into_vec();
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
                // We skipped the first centroid in the for loop
                closest_index = index + 1;
            }
        }

        closest_index
    }
}

struct RollingMean {
    pub current_mean: Array1<f64>,
    n_samples: u64,
}

impl RollingMean {
    pub fn new(first_sample: Array1<f64>) -> Self {
        RollingMean {
            current_mean: first_sample,
            n_samples: 1,
        }
    }

    pub fn accumulate<A>(&mut self, new_sample: &ArrayBase<A, Ix1>)
    where
        A: Data<Elem = f64>,
    {
        let mut increment: Array1<f64> = &self.current_mean - new_sample;
        increment.mapv_inplace(|x| x / (self.n_samples + 1) as f64);
        self.current_mean -= &increment;
        self.n_samples += 1;
    }
}
