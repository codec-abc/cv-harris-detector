// see 
// https://res.mdpi.com/d_attachment/sensors/sensors-10-02027/article_deploy/sensors-10-02027.pdf
//http://ir.hfcas.ac.cn:8080/bitstream/334002/31604/1/Automatic%20chessboard%20corner%20detection%20method.pdf

use image::{ImageBuffer, Luma};
use std::collections::HashMap;

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
    // Basically, we check for black and white alternating around the corner.

    // For example a valid corner would look like this: (B is black and W is white)
    // | W | B | or | B | W |
    // | B | W |    | W | B |

    // while an invalid corner might look like this:
    // | W | W |
    // | B | W |

    // This is the case for the chessboard outer corner for example.
    // This way, they are filtered out and we only keep the inner corners.

    let width = image.width();
    let height = image.height();

    // Get all 8 pixels intensity around the current corner

    // | i4 | i3 | i2 | 
    // | i5 | __ | i1 | 
    // | i6 | i7 | i8 |

    let i1: f64 = image[get_pixel_coord((x_c + 1, y_c + 0), width, height)][0] as f64;
    let i2: f64 = image[get_pixel_coord((x_c + 1, y_c + 1), width, height)][0] as f64;
    let i3: f64 = image[get_pixel_coord((x_c + 0, y_c + 1), width, height)][0] as f64;
    let i4: f64 = image[get_pixel_coord((x_c - 1, y_c + 1), width, height)][0] as f64;
    let i5: f64 = image[get_pixel_coord((x_c - 1, y_c + 0), width, height)][0] as f64;
    let i6: f64 = image[get_pixel_coord((x_c - 1, y_c - 1), width, height)][0] as f64;
    let i7: f64 = image[get_pixel_coord((x_c + 0, y_c - 1), width, height)][0] as f64;
    let i8: f64 = image[get_pixel_coord((x_c + 1, y_c - 1), width, height)][0] as f64;

    // Compute differences.

    let d1: f64 = (i1 - i5).abs();
    let d2: f64 = (i3 - i7).abs();
    let d3: f64 = (i1 + i5 - i3 - i7).abs();
    let d4: f64 = (i2 - i6).abs();
    let d5: f64 = (i4 - i8).abs();
    let d6: f64 = (i2 + i6 - i4 - i8).abs();

    // check if criterion is met
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
    // Basically, if a corner does not have at least 3 neighbors close enough it is not valid

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

        let distance = distance((self_x_f64, self_y_f64), (other_x_f64, other_y_f64));

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

pub fn apply_neighbor_angle_filter(
    t_cosine_threshold: f64,
    corners: &[CornerLocation],
    corner_index_to_check: usize
) -> CornerFilterResult 
{
    // see Section 3.2.3 Angle property:
    // Basically, the 2 most neighbors of a valid corners should make an a somewhat big angle.
    // Even with perspective.

    // Can't have 2 neighbors if we have not at least 3 corners in total.
    assert!(corners.len() >= 3);

    let (self_x, self_y) = corners[corner_index_to_check];
    let self_x_f64 = self_x as f64;
    let self_y_f64 = self_y as f64;

    // We store distance to a neighbor along with the neighbor's coordinates
    let mut closer_neighbor_1 = (std::f64::MAX, (0f64, 0f64));
    let mut closer_neighbor_2 = (std::f64::MAX, (0f64, 0f64));

    // Iterate over all corners to find the 2 closest
    for index in 0..corners.len() {

        // Don't make sense to check if pixel is close with itself
        if index == corner_index_to_check {
            continue;
        }

        let (other_x, other_y) = corners[index];

        let other_x_f64 = other_x as f64;
        let other_y_f64 = other_y as f64;

        let distance = distance((self_x_f64, self_y_f64), (other_x_f64, other_y_f64));

        let mut current_neighbor = (distance, (other_x_f64, other_y_f64));

        if current_neighbor.0 < closer_neighbor_1.0 {
            std::mem::swap(&mut current_neighbor, &mut closer_neighbor_1);
        }

        if current_neighbor.0 < closer_neighbor_2.0 {
            std::mem::swap(&mut current_neighbor, &mut closer_neighbor_2);
        }
    }

    // now calculates angles with the 2 closest neighbors.
    // ie, closer_neighbor_1 -> current_point -> closer_neighbor_2

    let (closer_neighbor_1_x, closer_neighbor_1_y) = closer_neighbor_1.1;
    let (closer_neighbor_2_x, closer_neighbor_2_y) = closer_neighbor_2.1;

    let a = ((self_x_f64  - closer_neighbor_1_x), (self_y_f64 - closer_neighbor_1_y));
    let b = ((self_x_f64  - closer_neighbor_2_x), (self_y_f64 - closer_neighbor_2_y));

    let cos_theta = (a.0 * b.0 + a.1 * b.1) / (norm(a) * norm(b));

    if cos_theta < t_cosine_threshold {
        CornerFilterResult::RealCorner
    } else {
        CornerFilterResult::FakeCorner
    }
}

fn distance((a_x, a_y) : (f64, f64), (b_x, b_y) : (f64, f64)) -> f64 {
    ((a_x - b_x).powi(2) + (a_y - b_y).powi(2)).sqrt()
}

fn norm((a_x, a_y) : (f64, f64)) -> f64 {
    (a_x.powi(2) + a_y.powi(2)).sqrt()
}

// return the index of the corners
fn get_corners(corners: &[CornerLocation]) -> (usize, usize, usize, usize) {
    let mut a = std::i64::MAX; // top-left corner
    let mut b = std::i64::MIN; // top-right corner
    let mut c = std::i64::MAX; // bottom-left corner
    let mut d = std::i64::MIN; // bottom-right corner

    let mut a_index = 0;
    let mut b_index = 0;
    let mut c_index = 0;
    let mut d_index = 0;

    // for all corners we sum and compute the diff between x and y.
    for index in 0..corners.len() {
        let (current_x, current_y) = corners[index];

        let sum = current_x as i64 + current_y as i64 ;
        let diff = current_x as i64 - current_y as i64 ;


        if sum < a {
            a = sum;
            a_index = index;
        }

        if diff > b {
            b = sum;
            b_index = index;
        }

        if diff < c {
            c = sum;
            c_index = index;
        }

        if sum > d {
            d = sum;
            d_index = index;
        }

    }

    (a_index, b_index, c_index, d_index)
}

struct chessboard_detector_parameters {
    r: f64,
    p: f64,
    d: f64,
    t: f64,
}

fn compute_adaptive_parameters(a_min: f64, a_max: f64) -> chessboard_detector_parameters {

    let r = 0.7f64 * a_min;
    let p = 0.3f64 * a_max / a_min;
    let d = 2.0f64 * a_max;
    let t = 0.4f64 * a_max / a_min;

    chessboard_detector_parameters { r, p, d, t}
}

struct DistanceHistogram {
    histogram: HashMap::<u32, u32>,
    number_of_values: u32,
    peak_index: u32,
}

fn compute_neighbor_distance_histogram(corners: &[CornerLocation]) -> DistanceHistogram {
    let mut histogram = HashMap::<u32, u32>::new();
    let mut sum = 0;

    for index_1 in 0..corners.len() {
        let (self_x, self_y) = corners[index_1];
        let self_x_f64 = self_x as f64;
        let self_y_f64 = self_y as f64;

        for index_2 in (index_1 + 1)..corners.len() {

            // Don't make sense to check if pixel with itself
            if index_1 == index_2 {
                continue;
            }

            let (other_x, other_y) = corners[index_2];

            let other_x_f64 = other_x as f64;
            let other_y_f64 = other_y as f64;

            let distance = distance((self_x_f64, self_y_f64), (other_x_f64, other_y_f64));
            let distance = distance as u32;

            sum += 1;
            *histogram.entry(distance).or_insert(0) += 1;

        }
    }

    let mut peak_index = 0;
    let mut peak_max_value = 0;

    for (key, value) in &histogram {
        if *value >= peak_max_value {
            peak_index = *key;
            peak_max_value = *value;
        }
    }

    DistanceHistogram {
        histogram: histogram,
        number_of_values: sum,
        peak_index: peak_index,
    }
}