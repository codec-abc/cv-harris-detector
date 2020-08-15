use image::{DynamicImage, Rgba};
use imageproc::{drawing};

// Run with cargo run --bin cv-harris-detector

pub fn main_harris() {
    //let image_path = "./cv-harris-detector/test_images/Harris_Detector_Original_Image.jpg";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";
    let image_path = "./cv-harris-detector/test_images/Calibim1.tif";

    let src_image = image::open(image_path).expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    // and https://docs.rs/image/0.23.8/src/image/color.rs.html#415

    let gray_image = src_image.to_luma();
    let width = gray_image.width();
    let height = gray_image.height();

    let k: f64 = 0.04f64; // Harris detector free parameter. The higher the value the less it detects.
    let blur = None;//Some(0.5f32); // TODO: fix this. For very low value the image starts to be completely white

    let harris_result = cv_harris_detector::harris_corner(&gray_image, k, blur);

    let harris_normed = cv_harris_detector::min_max_normalize_harris(&harris_result);

    let display_over_original_image = false;
    let threshold = 120;

    let mut canvas = if display_over_original_image {
        drawing::Blend(src_image.to_rgba())
    } else {
        drawing::Blend(harris_normed.clone())
    };

    for x in 0..width {
        for y in 0..height {
            let normed = harris_normed[(x, y)][0];

            if normed > threshold {
                drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x as i32, y as i32);
            }
            
        }
    }

    let out_img = DynamicImage::ImageRgba8(canvas.0);
    imgshow::imgshow(&out_img);
}

pub fn main_chessboard_detector() {
    let image_path = "./cv-harris-detector/test_images/Calibim1.tif";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";

    let src_image = image::open(image_path).expect("failed to open image file");

    let gray_image = src_image.to_luma();
    let width = gray_image.width();
    let height = gray_image.height();

    let mut canvas = drawing::Blend(src_image.to_rgba());


    let out_img = DynamicImage::ImageRgba8(canvas.0);
    imgshow::imgshow(&out_img);
}


pub fn main() {
    main_harris();
}