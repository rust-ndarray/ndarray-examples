use ndarray::prelude::*;
use ndarray::Data;

pub fn precision<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> f64
    where D: Data<Elem = f64> {
    let num_positive = x.iter().filter(|a| **a > 0.0).count() as f64;
    let num_true_positives = x.into_iter().zip(y.into_iter())
        .filter(|(a,b)| **a > 0.0 && **b)
        .count() as f64;

    num_true_positives / num_positive
}

pub fn accuracy<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> f64
    where D: Data<Elem = f64> {

    let num_correctly_classified = x.into_iter().zip(y.into_iter())
        .filter(|(a,b)| **a > 0.0 && **b || **a <= 0.0 && !**b)
        .count() as f64;

    let total_number = y.len() as f64;

    num_correctly_classified / total_number
}

pub fn recall<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> f64
    where D: Data<Elem = f64>
{
    let num_true_positives = x.into_iter().zip(y.into_iter())
        .filter(|(a,b)| **a > 0.0 && **b)
        .count() as f64;

    let total_number_positives = y.iter().filter(|x| **x).count() as f64;

    num_true_positives / total_number_positives
}

pub fn f1_score<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> f64
    where D: Data<Elem = f64>
{
    let recall = recall(x, y);
    let precision = precision(x, y);

    2.0 * (recall * precision) / (recall + precision)
}
