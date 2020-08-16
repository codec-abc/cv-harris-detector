use image::{DynamicImage, Rgb, imageops::FilterType};
use imageproc::{drawing};

use cv_harris_detector::*;

// Run with cargo run --bin cv-harris-detector

pub fn main_harris() {
    //let image_path = "./cv-harris-detector/test_images/Harris_Detector_Original_Image.jpg";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";
    let image_path = "./cv-harris-detector/test_images/bouguet/Image1.tif";

    let src_image = image::open(image_path).expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    // and https://docs.rs/image/0.23.8/src/image/color.rs.html#415

    let display_over_original_image = true;
    let harris_threshold = 20;
    let contrast_threshold = 120;
    let non_maximum_suppression_radius = 6;
    let window_size_ratio = 0.5f64;// TODO : should be 0.8;
    let k: f64 = 0.04f64; // Harris detector free parameter. The higher the value the less it detects.
    //let blur = None; //Some(0.3f32); // TODO: fix this. For very low value the image starts to be completely white

    let gray_image = src_image.to_luma();
    let width = gray_image.width();
    let height = gray_image.height();

    // resize-start
    let scale_ratio = 1;

    let gray_image = 
        src_image.resize(
            width / scale_ratio, 
            height / scale_ratio, 
            FilterType::Lanczos3).to_luma();

    let width = gray_image.width();
    let height = gray_image.height();
    // resize-end

    //let gray_image = imageproc::contrast::equalize_histogram(&gray_image);
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
                
                // drawing::draw_filled_circle_mut(
                //     &mut canvas, 
                //     (x as i32 , y as i32),
                //     1i32,
                //     Rgb([255, 0, 0]));

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

    let chessboard_parameters = compute_adaptive_parameters(a_min, a_max);

    println!("p is {}", chessboard_parameters.p);
    println!("d is {}", chessboard_parameters.d);
    println!("t is {}", chessboard_parameters.t);

    let mut wrong_corners_indexes: Vec<usize> = vec!();

    for (index, (x, y)) in corners.iter().enumerate()  {
        let corner_location = (*x, *y); 

        // let corner_filter = 
        //     apply_center_symmetry_filter(
        //         chessboard_parameters.p,
        //         &harris_normed_non_max_suppressed,
        //         corner_location
        //     );

        let corner_filter = apply_neighbor_distance_filter(
            chessboard_parameters.d,
            &corners,
            index,
        );

        // let corner_filter = apply_neighbor_angle_filter(
        //     chessboard_parameters.t,
        //     &corners,
        //     index,
        // );

        if corner_filter == CornerFilterResult::FakeCorner {
            println!("we have got a fake corner");
            wrong_corners_indexes.push(index);
        }
        
    }
    
    for (index, (x, y)) in corners.iter().enumerate()  {
        let is_fake_corner = wrong_corners_indexes.contains(&index);
        if is_fake_corner { 
            drawing::draw_filled_circle_mut(
                &mut canvas, 
                (*x as i32 , *y as i32),
                1i32,
                Rgb([255, 0, 0]));
        }
    }

    let out_img = DynamicImage::ImageRgb8(canvas.0);
    imgshow::imgshow(&out_img);
}


pub fn main() {
    main_harris();
}

// TODO: Non maximum suppression for Harris detector?