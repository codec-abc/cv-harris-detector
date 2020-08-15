// see 
// https://res.mdpi.com/d_attachment/sensors/sensors-10-02027/article_deploy/sensors-10-02027.pdf
//http://ir.hfcas.ac.cn:8080/bitstream/334002/31604/1/Automatic%20chessboard%20corner%20detection%20method.pdf

use image::{ImageBuffer, Luma, Rgba};

type GreyImage = ImageBuffer<Luma<u8>, Vec<u8>>;
type CornerLocation = (i32, i32);

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Clone)]
pub enum CornerFilterResult {
    FakeCorner,
    RealCorner
}

// call this for every corners
pub fn apply_center_symmetry_filter(
    p_threshold : f64, // should be in [0; 1] range
    image: &GreyImage, 
    (x_c, y_c): CornerLocation
) -> CornerFilterResult {

    // see Section 3.2.1 Centrosymmetry property

    let width = image.width();
    let height = image.height();

    // Get all 8 pixels intensity around the current corner
    let i1: f64 = image[get_pixel_coord((x_c + 1, y_c + 0), width, height)][0] as f64;
    let i2: f64 = image[get_pixel_coord((x_c + 1, y_c + 1), width, height)][0] as f64;
    let i3: f64 = image[get_pixel_coord((x_c + 0, y_c + 1), width, height)][0] as f64;
    let i4: f64 = image[get_pixel_coord((x_c - 1, y_c + 1), width, height)][0] as f64;
    let i5: f64 = image[get_pixel_coord((x_c - 1, y_c + 0), width, height)][0] as f64;
    let i6: f64 = image[get_pixel_coord((x_c - 1, y_c - 1), width, height)][0] as f64;
    let i7: f64 = image[get_pixel_coord((x_c + 0, y_c - 1), width, height)][0] as f64;
    let i8: f64 = image[get_pixel_coord((x_c + 1, y_c - 1), width, height)][0] as f64;

    let d1: f64 = (i1 - i5).abs();
    let d2: f64 = (i3 - i7).abs();
    let d3: f64 = (i1 + i5 - i3 - i7).abs();
    let d4: f64 = (i2 - i6).abs();
    let d5: f64 = (i4 - i8).abs();
    let d6: f64 = (i2 + i6 - i4 - i8).abs();

    let is_real_corner =
        (d1 < p_threshold * d3 && d2 < p_threshold * d3) ||
        (d4 < p_threshold * d6 && d5 < p_threshold * d6);

    if is_real_corner {
        CornerFilterResult::RealCorner
    } else {
        CornerFilterResult::FakeCorner
    }
}

// clamp pixel coordinates. It should be better do to something like OpenCV does with BorderTypes
// example: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_BorderTypes.html
pub fn get_pixel_coord((x, y): CornerLocation, width: u32, height: u32) -> (u32, u32) {
    let mut x_o = x;
    let mut y_o = y;

    if x_o < 0 {
        x_o = 0;
    }

    if y_o < 0 {
        y_o = 0;
    }

    if x_o > width as i32 - 1i32 {
        x_o = width as i32 - 1i32;
    }

    if y_o > height as i32 - 1i32 {
        y_o = height as i32 - 1i32;
    }

    (x_o as u32, y_o as u32)
}

pub fn apply_neighbor_distance_filter(
    distance_threshold: f64,
    corners: &[CornerLocation],
    corner_index_to_check: usize
) -> CornerFilterResult  
{
    // see 3.2.2 Distance property:

    let (self_x, self_y) = corners[corner_index_to_check];
    let self_x_f64 = self_x as f64;
    let self_y_f64 = self_y as f64;

    let mut neighbor_count = 0;
    
    for index in 0..corners.len() {

        // Don't make sense to check if pixel is close with itself
        if index == corner_index_to_check {
            continue;
        }

        let (other_x, other_y) = corners[index];

        let other_x_f64 = other_x as f64;
        let other_y_f64 = other_y as f64;

        let distance = ((self_x_f64 - other_x_f64).powi(2) + (self_y_f64 - other_y_f64).powi(2)).sqrt();

        if distance <= distance_threshold {
            neighbor_count = neighbor_count + 1;
        }
    }

    if neighbor_count >= 3 {
        CornerFilterResult::RealCorner
    } else {
        CornerFilterResult::FakeCorner
    }
}

pub fn apply_neighbor_angle_filter() {
    todo!();

    // see Section 3.5 in Automatic chessboard corner detection method
}