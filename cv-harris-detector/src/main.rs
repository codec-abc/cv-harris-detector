use image::{DynamicImage, ImageBuffer, Luma}; // GenericImageView Rgba
use imageproc::{drawing, gradients};

// Run with cargo run --bin cv-harris-detector


// see 
// http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
// https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6
// https://en.wikipedia.org/wiki/Harris_Corner_Detector
// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
// https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris
// https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/corner.cpp
// https://github.com/codeplaysoftware/visioncpp/wiki/Example:-Harris-Corner-Detection

pub fn main() {
    let src_image = 
        image::open("./cv-harris-detector/test_images/fileListImageUnDist.jpg")
        .expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray


    let grey_image = src_image.to_luma();

    let block_size : u32 = 2u32; // Neighborhood size (see the details on cornerEigenValsAndVecs() ).
    let k_size : u32 = 3u32; // Aperture parameter for the Sobel() operator.
    let k : f64 = 0.04f64;  // Harris detector free parameter. See the formula below

    harris_corner(grey_image, block_size, k_size, k);
    
    let mut canvas = drawing::Blend(src_image.to_luma());

    //let x = 12i32;
    //let y = 12i32;
    //drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x as i32, y as i32);
    
    let out_img = DynamicImage::ImageLuma8(canvas.0);
    imgshow::imgshow(&out_img);
}

#[allow(unused_variables)]
fn harris_corner(grey_image: ImageBuffer<Luma<u8>, Vec<u8>>, block_size : u32, k_size : u32, k : f64) -> () {

    let sobel_horizontal = gradients::horizontal_sobel(&grey_image);
    let sobel_vertical = gradients::vertical_sobel(&grey_image);

    let width = grey_image.width();
    let height = grey_image.height();

    let mut i_x2_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_y2_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_xy_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);

    for x in 0..grey_image.width() - 1 {
        for y in 0..grey_image.height() -1 {
            let i_x = sobel_horizontal[(x, y)][0] as f64;
            let i_y = sobel_vertical[(x, y)][0] as f64;
            let i_x2 = i_x * i_x;
            let i_y2 = i_y * i_y;
            let i_xy = i_x * i_y;

            i_x2_image[(x, y)] = Luma::from([i_x2]);
            i_y2_image[(x, y)] = Luma::from([i_y2]);
            i_xy_image[(x, y)] = Luma::from([i_xy]);
        }
    }

    let kernel : Vec<f64> = vec![1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64];
    let i_x2_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_x2_image, &kernel);
    let i_y2_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_y2_image, &kernel);
    let i_xy_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_xy_image, &kernel);
}

