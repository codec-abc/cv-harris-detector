use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgba};
use imageproc::{filter, gradients, drawing};

// Run with cargo run --bin cv-harris-detector

// see
// http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
// https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6
// https://en.wikipedia.org/wiki/Harris_Corner_Detector
// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
// https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris
// https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/corner.cpp
// https://github.com/codeplaysoftware/visioncpp/wiki/Example:-Harris-Corner-Detection

pub struct HarrisDetectorResult {
    pub detector_result: ImageBuffer<Luma<f64>, Vec<f64>>,
    pub min: f64,
    pub max: f64,
}

pub fn main() {
    let image_path = "./cv-harris-detector/test_images/Harris_Detector_Original_Image.jpg";
    //let image_path = "./cv-harris-detector/test_images/fileListImageUnDist.jpg";

    let src_image = image::open(image_path).expect("failed to open image file");

    // Probably not the right kind of conversion
    // see https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
    // and https://docs.rs/image/0.23.8/src/image/color.rs.html#415

    let gray_image = src_image.to_luma();
    let width = gray_image.width();
    let height = gray_image.height();

    let k: f64 = 0.04f64; // Harris detector free parameter. The higher the value the less it detects.
    let blur = None;//Some(0.5f32); // TODO: fix this. For very low value the image starts to be completely white

    let harris_result = harris_corner(&gray_image, k, blur);
    let min = harris_result.min;
    let max = harris_result.max;

    let threshold = 200f64;

    let mut canvas = drawing::Blend(src_image.to_rgba());

    for x in 0..width {
        for y in 0..height {
            let harris_val_f64 = harris_result.detector_result[(x, y)][0];

            // min max normlization
            let a = 0.0f64;
            let b = 255.0f64;

            let normed = a + ((harris_val_f64 - min) * (b - a)) / (max - min);

            if normed > threshold {
                drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x as i32, y as i32);
            }
            
        }
    }

    let out_img = DynamicImage::ImageRgba8(canvas.0);
    imgshow::imgshow(&out_img);
}

pub fn harris_corner(
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    k: f64,
    blur: Option<f32>,
) -> HarrisDetectorResult {
    let blurred_image: Option<ImageBuffer<Luma<u8>, Vec<u8>>>;

    let gray_image: &ImageBuffer<Luma<u8>, Vec<u8>> = match blur {
        Some(f) => {
            blurred_image = Some(filter::gaussian_blur_f32(&gray_image, f));
            blurred_image.as_ref().unwrap()
        }
        None => gray_image,
    };

    let sobel_horizontal = gradients::horizontal_sobel(&gray_image);
    let sobel_vertical = gradients::vertical_sobel(&gray_image);

    let width = gray_image.width();
    let height = gray_image.height();

    let mut i_x2_image: ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_y2_image: ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);
    let mut i_xy_image: ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);

    for x in 0..width {
        for y in 0..height {
            let i_x = sobel_horizontal[(x, y)][0] as f64 / 255.0f64;
            let i_y = sobel_vertical[(x, y)][0] as f64 / 255.0f64;
            let i_x2 = i_x * i_x;
            let i_y2 = i_y * i_y;
            let i_xy = i_x * i_y;

            i_x2_image[(x, y)] = Luma::from([i_x2]);
            i_y2_image[(x, y)] = Luma::from([i_y2]);
            i_xy_image[(x, y)] = Luma::from([i_xy]);
        }
    }

    let kernel: Vec<f64> = vec![
        1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64, 1.0f64,
    ];

    let i_x2_sum: ImageBuffer<Luma<f64>, Vec<f64>> =
        imageproc::filter::filter3x3(&i_x2_image, &kernel);

    let i_y2_sum: ImageBuffer<Luma<f64>, Vec<f64>> =
        imageproc::filter::filter3x3(&i_y2_image, &kernel);
        
    let i_xy_sum: ImageBuffer<Luma<f64>, Vec<f64>> =
        imageproc::filter::filter3x3(&i_xy_image, &kernel);

    let mut harris: ImageBuffer<Luma<f64>, Vec<f64>> = ImageBuffer::new(width, height);

    let mut harris_min = std::f64::MAX;
    let mut harris_max = std::f64::MIN;

    for x in 0..width {
        for y in 0..height {
            let ksumpx2 = i_x2_sum[(x, y)][0] as f64;
            let ksumpy2 = i_y2_sum[(x, y)][0] as f64;
            let ksumpxy = i_xy_sum[(x, y)][0] as f64;

            let mul1 = ksumpx2 * ksumpy2;
            let mul2 = ksumpxy * ksumpxy;
            let det = mul1 - mul2;

            let trace = ksumpx2 + ksumpy2;
            let trace2 = trace * trace;

            let ktrace2 = trace2 * k;

            let harris_val = det - ktrace2;

            harris_min = if harris_val < harris_min { harris_val } else { harris_min };
            harris_max = if harris_val > harris_max { harris_val } else { harris_max };

            harris[(x, y)] = Luma::from([harris_val]);
        }
    }

    HarrisDetectorResult {
        detector_result: harris,
        min: harris_min,
        max: harris_max,
    }
    
}
