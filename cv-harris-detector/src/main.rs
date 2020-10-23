use image::{DynamicImage, ImageBuffer, Rgb};
use imageproc::{drawing::{self, Blend}, filter};

use cv_harris_detector::*;

// Run with cargo run --bin cv-harris-detector

fn get_image_path_and_chessboard_size() -> (String, (i32, i32)) {
    //let image_path = "./cv-harris-detector/test_images/Harris_Detector_Original_Image.jpg";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";
    //let image_path = "./cv-harris-detector/test_images/bouguet/Image5.tif";
    //let image_path = "./cv-harris-detector/test_images/other/CalibIm1.tif";
    let image_path = "./cv-harris-detector/test_images/stereopi-tutorial/left_05.png"; // 9x6

    let chessboard_size = (9, 6);
    (image_path.to_owned(), chessboard_size)
}

fn generate_chessboard_parameters(
    mean: f64,
    std_dev: f64,
    window_size: u32,
    closest_neighbor_distance_histogram: &ClosestNeighborDistanceHistogram,
) -> ChessboardDetectorParameters {
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

    chessboard_parameters
}

pub fn main_harris() {

    let (image_path, _chessboard_size) = get_image_path_and_chessboard_size();
    let src_image = image::open(image_path).expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    // and https://docs.rs/image/0.23.8/src/image/color.rs.html#415

    let display_over_original_image = true;
    let harris_threshold = 70;
    
    let non_maximum_suppression_radius = 5.0f64;
    let window_size_ratio = 0.5f64;// TODO : should be 0.8;
    let run_histogram_normalization = true;
    let k: f64 = 0.04f64; // Harris detector free parameter. The higher the value the less it detects.

    let gray_image = src_image.to_luma();

    let blurred_gray_image = filter::gaussian_blur_f32(&gray_image, 2.0f32);

    // TODO : do local normalization instead of the whole image to cope with contrast differences
    let gray_image = if run_histogram_normalization {
        let old_gray_image = gray_image;
        //imageproc::contrast::equalize_histogram(&old_gray_image)
        let otsu_level = imageproc::contrast::otsu_level(&old_gray_image);
        imageproc::contrast::threshold(&old_gray_image, otsu_level)
    } else {
        gray_image
    };

    let harris_result = cv_harris_detector::harris_corner(&gray_image, k);

    let harris_normed_non_max_suppressed = 
        harris_result.run_non_maximum_suppression(non_maximum_suppression_radius, harris_threshold);

    let harris_image = DynamicImage::ImageLuma8(harris_normed_non_max_suppressed.clone()).to_rgb();

    let corners = get_harris_corners_based_on_threshold(&harris_normed_non_max_suppressed, harris_threshold);

    let number_of_corners = corners.len();

    println!("detected {} corners", number_of_corners);
    println!("we should have roughly twice as corners as there is in the pattern");

    let closest_neighbor_distance_histogram = compute_closest_neighbor_distance_histogram(&corners);
    let window_size = closest_neighbor_distance_histogram.window_size_that_cover_x_percent(window_size_ratio); 
    let (mean, std_dev) = closest_neighbor_distance_histogram.mean_val_and_std_dev_for_window(window_size);

    let chessboard_parameters = generate_chessboard_parameters(mean, std_dev, window_size, &closest_neighbor_distance_histogram);
    let filtering_result = filter_out_corners(&chessboard_parameters, &corners, &blurred_gray_image);
    
    println!("remaining corners {}", filtering_result.remaining_corners.len());
    println!("filtered out corners {}", filtering_result.filtered_out_corners.len());

    //let draw_eliminated = false;
    //draw_filtering_result(&mut canvas, draw_eliminated, &filtering_result);

    let corners_centers = find_corners_mean_and_medium(&filtering_result.remaining_corners);

    // drawing::draw_filled_circle_mut(
    //     &mut canvas, 
    //     corners_centers.mean,
    //     1i32,
    //     Rgb([255, 0, 0]));

    // drawing::draw_filled_circle_mut(
    //     &mut canvas, 
    //     corners_centers.medium,
    //     1i32,
    //     Rgb([0, 255, 0]));

    // let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    // imgshow::imgshow(&out_img);

    run_chessboard_detection(
        &filtering_result.remaining_corners, 
        &corners_centers,
        &blurred_gray_image
    );
    
}


pub fn main() {
    main_harris();
}