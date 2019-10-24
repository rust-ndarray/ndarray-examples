#![allow(non_snake_case)]
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};

use ndarray::{Array, Array2, Axis};
use ndarray_stats::QuantileExt;

use svm::{SupportVectorMachine, scores};

/// Read in dataset
fn dataset<T: AsRef<Path> + Copy>(path: T) -> (Array2<f64>, Vec<bool>) {
    let num_lines = BufReader::new(File::open(path).unwrap()).lines().count();
    let mut x = Array::zeros((num_lines, 20));
    let mut y = Vec::with_capacity(num_lines);

    for (i1, line) in BufReader::new(File::open(path).unwrap()).lines().map(|x| x.unwrap()).enumerate() {
        let mut iter = line.split(",").skip(1);
        y.push(iter.next().unwrap() == "M");

        for (i2, elm) in iter.map(|x| x.parse::<f64>().unwrap()).take(20).enumerate() {
            x[(i1,i2)] = elm;
        }
    }

    for mut col in x.axis_iter_mut(Axis(1)) {
        let min = *col.min().unwrap();
        let max = *col.max().unwrap();
        col -= min;
        col /= max - min;
    }

    (x, y)
}

fn main() {
    let (X,y) = dataset("./src/wdbc.data");

    let split_idx = (X.dim().0 as f64 * 0.9).floor() as usize;
    let (training_x, testing_x) = X.view().split_at(Axis(0), split_idx);
    let (training_y, testing_y) = y.split_at(split_idx);

    let mut svm = SupportVectorMachine::new();
    svm.fit(&training_x, &training_y);

    // calculate precision
    let prediction = svm.predict(&testing_x);

    println!("Accuracy {}, Precision {}, Recall {}, F1 score {}", 
             scores::accuracy(&prediction, &testing_y),
             scores::precision(&prediction, &testing_y),
             scores::recall(&prediction, &testing_y),
             scores::f1_score(&prediction, &testing_y));
}
