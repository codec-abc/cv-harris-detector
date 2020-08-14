use image::{DynamicImage, ImageBuffer, Luma, GrayImage}; // GenericImageView Rgba
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
    //let k : f64 = 0.04f64;  // Harris detector free parameter. See the formula below
    let k : f64 = 0.04f64;

    harris_corner(grey_image, k); // block_size, k_size,
    
    let mut canvas = drawing::Blend(src_image.to_luma());

    //let x = 12i32;
    //let y = 12i32;
    //drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x as i32, y as i32);
    
    let out_img = DynamicImage::ImageLuma8(canvas.0);
    imgshow::imgshow(&out_img);
}

#[allow(unused_variables)]
fn harris_corner(grey_image: ImageBuffer<Luma<u8>, Vec<u8>>, k : f64) -> () { //  block_size : u32, k_size : u32

    let sobel_horizontal = gradients::horizontal_sobel(&grey_image);
    let sobel_vertical = gradients::vertical_sobel(&grey_image);

    let width = grey_image.width();
    let height = grey_image.height();

    let mut i_x2_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_y2_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_xy_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);

    let mut min_ix = f64::MAX;
    let mut max_ix = f64::MIN;

    let mut min_iy = f64::MAX;
    let mut max_iy = f64::MIN;

    let mut min_ix2 = f64::MAX;
    let mut max_ix2 = f64::MIN;

    let mut min_iy2 = f64::MAX;
    let mut max_iy2 = f64::MIN;

    let mut min_ixy = f64::MAX;
    let mut max_ixy = f64::MIN;


    for x in 2..grey_image.width() - 2 {
        for y in 2..grey_image.height() - 2 {
            let i_x = sobel_horizontal[(x, y)][0] as f64 / 255.0f64;
            let i_y = sobel_vertical[(x, y)][0] as f64 / 255.0f64;
            let i_x2 = i_x * i_x;
            let i_y2 = i_y * i_y;
            let i_xy = i_x * i_y;

            i_x2_image[(x, y)] = Luma::from([i_x2]);
            i_y2_image[(x, y)] = Luma::from([i_y2]);
            i_xy_image[(x, y)] = Luma::from([i_xy]);


            if min_ix > i_x {
                min_ix = i_x;
            }

            if max_ix < i_x {
                max_ix = i_x;
            }

            if min_iy > i_y {
                min_iy = i_y;
            }

            if max_iy < i_y {
                max_iy = i_y;
            }


            if min_ix2 > i_x2 {
                min_ix2 = i_x2;
            }

            if max_ix2 < i_x2 {
                max_ix2 = i_x2;
            }

            if min_iy2 > i_y2 {
                min_iy2 = i_y2;
            }

            if max_iy2 < i_y2 {
                max_iy2 = i_y2;
            }

            if min_ixy > i_xy {
                min_ixy = i_xy;
            }

            if max_ixy < i_xy {
                max_ixy = i_xy;
            }
        }
    }

    println!("min_ix is {}", min_ix);
    println!("max_ix is {}", max_ix);

    println!("min_iy is {}", min_iy);
    println!("max_iy is {}", max_iy);

    println!("min_ix2 is {}", min_ix2);
    println!("max_ix2 is {}", max_ix2);

    println!("min_iy2 is {}", min_ix2);
    println!("max_iy2 is {}", max_iy2);

    println!("min_ixy is {}", min_ixy);
    println!("max_ixy is {}", max_ixy);


    let kernel : Vec<f64> = vec![1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64];
    let i_x2_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_x2_image, &kernel);
    let i_y2_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_y2_image, &kernel);
    let i_xy_sum: ImageBuffer<Luma<f64>, Vec<f64>> = imageproc::filter::filter3x3(&i_xy_image, &kernel);

    let mut min = f64::MAX;
    let mut max = f64::MIN;

    let mut debug_image : ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);

    for x in 4..grey_image.width() - 4 {
        for y in 4..grey_image.height() - 4 {
            let ksumpx2 = i_x2_sum[(x, y)][0] as f64;
            let ksumpy2 = i_y2_sum[(x, y)][0] as f64;
            let ksumpxy = i_xy_sum[(x, y)][0] as f64;

            let mul1 = ksumpx2 * ksumpy2;
            let mul2 = ksumpxy * ksumpxy;
            let det = mul1 - mul2;

            let trace = ksumpx2 + ksumpy2;
            let trace2 = trace * trace;

            let k_node = k;
            let ktrace2 = trace2 * k_node;

            let harris = det - ktrace2;

            if min > harris {
                min = harris;
            }

            if max < harris {
                max = harris;
            }

            debug_image[(x, y)] = Luma::from([harris]);
        }
    }

    println!("min is {}", min);
    println!("max is {}", max);

    let mut img = GrayImage::new(width, height);

    for x in 0..grey_image.width() - 1 {
        for y in 0..grey_image.height() -1 {
            //let value = sobel_vertical[(x, y)][0] as f64 / 1.0f64;
            //let value = sobel_horizontal[(x, y)][0] as f64 / 1.0f64;
            //let value = i_xy_image[(x, y)][0] as f64 / 4.0f64;
            //let value = i_x2_image[(x, y)][0] as f64 / 4.0f64;
            let value = debug_image[(x, y)][0] as f64 / 1.0f64;
            img.put_pixel(x, y, Luma([value as u8]));
        }
    }

    let img = DynamicImage::ImageLuma8(img);
    imgshow::imgshow(&img);

}