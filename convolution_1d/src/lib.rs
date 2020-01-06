#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;

pub fn convolve(data: ArrayView1<f64>, window: ArrayView1<f64>) -> Array1<f64> {
    let padded = stack![
        Axis(0),
        Array1::zeros(window.len() / 2),
        data,
        Array1::zeros(window.len() / 2)
    ];
    let mut w = window.view();
    w.invert_axis(Axis(0));

    padded
        .windows(w.len())
        .into_iter()
        .map(|x| (&x * &w).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn convolve_odd_odd() {
        let data = array![1., 2., 3.];
        let window = array![0., 1., 0.5];
        let expected = array![1., 2.5, 4.];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view())) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_odd_odd2() {
        let data = array![1., 2., 3., 4., 5.];
        let window = array![2., 1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view());
        let expected = array![8., 12., 16.5, 9., 5.5];

        for (exp, res) in expected.iter().zip(&result) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_even_odd() {
        let data = array![1., 2., 3., 4.];
        let window = array![0., 1., 0.5];
        let expected = array![1., 2.5, 4., 5.5];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view())) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_even_even() {
        let data = array![1., 2., 3., 4.];
        let window = array![1., 0.5];
        let expected = array![1., 2.5, 4., 5.5];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view())) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_even_even2() {
        let data = array![1., 2., 3., 4.];
        let window = array![1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view());
        let expected = array![2., 4., 6.5, 4.];

        for (exp, res) in expected.iter().zip(&result) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_odd_even() {
        let data = array![1., 2., 3., 4., 5.];
        let window = array![1., 0.5];
        let expected = array![1., 2.5, 4., 5.5, 7.];

        for (exp, res) in expected.iter().zip(&convolve(data.view(), window.view())) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }

    #[test]
    fn convolve_bigger_window() {
        let data = array![1., 2., 3.];
        let window = array![1., 0., 1., 0.5];
        let result = convolve(data.view(), window.view());
        let expected = array![2., 4., 2.5, 4.];

        for (exp, res) in expected.iter().zip(&result) {
            assert!(approx_eq!(f64, *exp, *res, ulps = 2));
        }
    }
}
