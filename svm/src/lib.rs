#![allow(non_snake_case)]
use ndarray::{Array, Array1, ArrayBase, Data, Ix2};
use ndarray_stats::DeviationExt;
use std::iter::FromIterator;

pub mod scores;

pub struct SupportVectorMachine {
    normal: Option<Array1<f64>>,
    bias: Option<f64>
}

impl SupportVectorMachine {
    pub fn new() -> SupportVectorMachine {
        SupportVectorMachine {
            normal: None, bias: None
        }
    }

    pub fn fit<A>(&mut self, X: &ArrayBase<A, Ix2>, y_bool: &[bool])
        where A: Data<Elem = f64>,
    {
        let (n_samples, _) = X.dim();
        let y = Array::from_iter(y_bool.into_iter().map(|x| if *x { 1.0 } else { -1.0 }));

        assert!(
            n_samples == y.dim(),
            "We need the same number of samples as well as targets!"
        );

        let mut multiplier = Array::ones(n_samples);

        loop {
            let tmp = (&y * &multiplier).dot(X);
            let gamma = &X.dot(&tmp) * &y;

            let update = 0.000000000001 * ( 1.0 - gamma );

            let mut new_multiplier = &multiplier + &update;
            new_multiplier.mapv_inplace(|x| f64::max(0.0, x));

            let distance = multiplier.sq_l2_dist(&new_multiplier).unwrap();
            multiplier = new_multiplier;

            println!("{}", distance);

            if distance < 1e-14 {
                break;
            }
        }

        let normal = (&y * &multiplier).dot(X);
        let z = &X.dot(&normal);

        // TODO pick min from positive and max from negative class
        let min = z.iter().zip(y_bool.iter()).filter(|(_,y)| **y).map(|(x,_)| *x).fold(0./0., f64::min);
        let max = z.iter().zip(y_bool.iter()).filter(|(_,y)| !**y).map(|(x,_)| *x).fold(0./0., f64::max);
        let bias = (min - max) / 2.0;

        self.normal = Some(normal);
        self.bias = Some(bias);
    }

    pub fn predict<A>(&self, X: &ArrayBase<A, Ix2>) -> Array1<f64>
        where A: Data<Elem = f64>
    {
        if let (Some(ref normal), Some(ref bias)) = (&self.normal, &self.bias) {
            let mut estimate = X.dot(normal);
            estimate.mapv_inplace(|x| x + bias);

            estimate
        } else {
            Array::zeros(X.dim().1)
        }
    }
}
