use ndarray::prelude::*;
use convolution_1d::convolve;

fn main() {
    println!(r#"numpy example of convolve with mode='same':
>>> np.convolve([1,2,3],[0,1,0.5], 'same')
array([1. ,  2.5,  4. ])

ndarray code follows:"#);

    let data = array![1., 2., 3.];
    let window = array![0., 1., 0.5];
    println!("data: \t\t{:?}", data);
    println!("window: \t{:?}", window);
    println!("convolution: \t{:?}", convolve(data.view(), window.view()));
}
