use image::{DynamicImage, ImageBuffer, Luma, Rgb};
use imageproc::{filter, gradients};

use crate::common::get_pixel_coord;

// see
// http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf
// https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6
// https://en.wikipedia.org/wiki/Harris_Corner_Detector
// https://docs.opencv.org/3.4/d4/d7d/tutorial_harris_detector.html
// https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris
// https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/corner.cpp
// https://github.com/codeplaysoftware/visioncpp/wiki/Example:-Harris-Corner-Detection

pub struct HarrisDetectorResult {
    detector_result: ImageBuffer<Luma<f64>, Vec<f64>>,
    min: f64, // minimum value computed by the Harris detector
    max: f64, // maximum value computed by the Harris detector
    width: u32, // image width
    height: u32, // image height
}

impl HarrisDetectorResult {
    pub fn get_image(&self) -> ImageBuffer<Luma<f64>, Vec<f64>> {
        self.detector_result.clone()
    }

    pub fn min_max_normalized_harris(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    
        let width = self.width;
        let height = self.height;
    
        let mut harris_normalized: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    
        let min = self.min;
        let max = self.max;
    
        for x in 0..width {
            for y in 0..height {
                let harris_val_f64 = self.detector_result[(x, y)][0];
    
                // min max normalization
                let a = 0.0f64;
                let b = 255.0f64;
    
                let normed = a + ((harris_val_f64 - min) * (b - a)) / (max - min);
                let normed_u8 = normed as u8;
                harris_normalized[(x, y)] = Rgb([normed_u8, normed_u8, normed_u8]);
                
            }
        }
    
        harris_normalized
    }

    pub fn run_non_maximum_suppression(
        &self,
        non_maximum_suppression_radius: u32,
        harris_threshold: u8
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut harris_normalized = DynamicImage::ImageRgb8(self.min_max_normalized_harris()).to_luma();

        let width = self.width;
        let height = self.height;

        for x in 0..width as i32 {
            for y in 0..height as i32 {
                let pixel_coord = (x as u32, y as u32);
                let value = harris_normalized[pixel_coord][0];

                let non_maximum_suppression_radius = non_maximum_suppression_radius as i32 + 1;

                for i in (-non_maximum_suppression_radius + 1)..non_maximum_suppression_radius {
                    for j in (-non_maximum_suppression_radius + 1)..non_maximum_suppression_radius {
                        if i == 0 && j == 0 {
                            continue;
                        }

                        let other_value = harris_normalized[get_pixel_coord((x + i, y + j), width, height)][0];

                        if value >= harris_threshold && value <= other_value {
                            harris_normalized[pixel_coord] = Luma([0]);
                        }
                    }
                }
            }
        }

        harris_normalized
    }
}

pub fn harris_corner(
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    k: f64,
    //blur: Option<f32>,
) -> HarrisDetectorResult {
    let blurred_image: Option<ImageBuffer<Luma<u8>, Vec<u8>>>;

    // TODO : fix blur, it doesn't work great
    // may use http://dev.theomader.com/gaussian-kernel-calculator/

    /*
    let gray_image: &ImageBuffer<Luma<u8>, Vec<u8>> = match blur {
        Some(f) => {
            blurred_image = Some(filter::gaussian_blur_f32(&gray_image, f));
            blurred_image.as_ref().unwrap()
        }
        None => gray_image,
    };
    */

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
        width: width,
        height: height,
    }
    
}

