use image::{DynamicImage, Rgb, imageops::FilterType};
use imageproc::{drawing, filter};

use cv_harris_detector::*;

// Run with cargo run --bin cv-harris-detector


pub fn main_harris() {
    //let image_path = "./cv-harris-detector/test_images/Harris_Detector_Original_Image.jpg";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";
    
    let image_path = "./cv-harris-detector/test_images/bouguet/Image5.tif";
    //let image_path = "./cv-harris-detector/test_images/other/CalibIm1.tif";

    let image_path = "./cv-harris-detector/test_images/stereopi-tutorial/left_01.png";

    let src_image = image::open(image_path).expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    // and https://docs.rs/image/0.23.8/src/image/color.rs.html#415

    let display_over_original_image = true;
    let harris_threshold = 70;
    
    let non_maximum_suppression_radius = 5.0f64;
    let window_size_ratio = 0.5f64;// TODO : should be 0.8;
    let run_histogram_normalization = false;
    let k: f64 = 0.04f64; // Harris detector free parameter. The higher the value the less it detects.
    //let blur = None; //Some(0.3f32); // TODO: fix this. For very low value the image starts to be completely white

    let gray_image = src_image.to_luma();
    let width = gray_image.width();
    let height = gray_image.height();

    let blurred_gray_image = filter::gaussian_blur_f32(&gray_image, 2.0f32);

    // TODO : do local normalization instead of the whole image to cope with contrast differences
    let gray_image = if run_histogram_normalization {
        let old_gray_image = gray_image;
        imageproc::contrast::equalize_histogram(&old_gray_image)
    } else {
        gray_image
    };

    //let contrast_threshold = 120;
    //let gray_image = imageproc::contrast::threshold(&gray_image, contrast_threshold);

    let harris_result = cv_harris_detector::harris_corner(&gray_image, k);// blur;

    let harris_normed_non_max_suppressed = 
        harris_result.run_non_maximum_suppression(non_maximum_suppression_radius, harris_threshold);

    let harris_image = DynamicImage::ImageLuma8(harris_normed_non_max_suppressed.clone()).to_rgb();

    let mut canvas = if display_over_original_image {
        drawing::Blend(DynamicImage::ImageLuma8(gray_image).to_rgb())
    } else {
        drawing::Blend(harris_image)
    };

    let mut corners = Vec::new();
    
    for x in 0..width {
        for y in 0..height {
            let normed = harris_normed_non_max_suppressed[(x, y)][0];

            if normed >= harris_threshold {
                corners.push((x as i32, y as i32));
            }
            
        }
    }

    let number_of_corners = corners.len();

    println!("detected {} corners", number_of_corners);
    println!("we should have roughly twice as corners as there is in the pattern");

    // TODO filter corners

    let closest_neighbor_distance_histogram = compute_closest_neighbor_distance_histogram(&corners);
    let window_size = closest_neighbor_distance_histogram.window_size_that_cover_x_percent(window_size_ratio); 
    let (mean, std_dev) = closest_neighbor_distance_histogram.mean_val_and_std_dev_for_window(window_size);

    println!(
        "peak index is {}, peak value is {}", 
        closest_neighbor_distance_histogram.get_peak_index(), 
        closest_neighbor_distance_histogram.get_peak_value()
    );

    println!("window size is {}", window_size);

    println!("mean is {} std_dev is {}", mean, std_dev);

    let a_min = mean - 3.0f64 * std_dev;
    let a_min = if a_min < 0f64 { 0.0f64 } else  { a_min };
    let a_max = mean + 3.0f64 * std_dev;

    println!("a_min is {}, a_max is {}", a_min, a_max);

    let mut chessboard_parameters = compute_adaptive_parameters(a_min, a_max);

    // TODO : clamp t to < -1
    if chessboard_parameters.t >= 1.0f64 {
        chessboard_parameters.t = 0.9f64;
    }

    // TODO : find good values for d
    chessboard_parameters.d = chessboard_parameters.d / 1.3f64;

    // TODO : find good values for p
    chessboard_parameters.p = 1.2f64;

    println!("p is {}", chessboard_parameters.p);
    println!("d is {}", chessboard_parameters.d);
    println!("t is {}", chessboard_parameters.t);

    let filtering_result = filter_out_corners(&chessboard_parameters, &corners, &blurred_gray_image);
    let draw_eliminated = false;

    println!("remaining corners {}", filtering_result.remaining_corners.len());
    println!("filtered out corners {}", filtering_result.filtered_out_corners.len());

    
    draw_filtering_result(&mut canvas, draw_eliminated, &filtering_result);
}


pub fn main() {
    main_harris();
}