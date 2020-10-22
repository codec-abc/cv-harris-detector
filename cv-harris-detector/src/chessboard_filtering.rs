// see 
// https://res.mdpi.com/d_attachment/sensors/sensors-10-02027/article_deploy/sensors-10-02027.pdf
// http://ir.hfcas.ac.cn:8080/bitstream/334002/31604/1/Automatic%20chessboard%20corner%20detection%20method.pdf
// https://web.stanford.edu/class/cs231a/prev_projects_2015/chess.pdf
// https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf
// https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Dekelaita.pdf
// http://rcv.kaist.ac.kr/?module=file&act=procFileDownload&file_srl=5272&sid=1a773ef2a2684da674c6825044658f27&module_srl=4502
// https://pdfs.semanticscholar.org/6e2d/9fa0c1a8c3798b30a0e5dc16f8ae85415c4e.pdf
// https://www.scitepress.org/papers/2017/61043/61043.pdf
// http://rpg.ifi.uzh.ch/docs/IROS08_scaramuzza_b.pdf
// https://www.graphicon.ru/oldgr/en/research/calibration/opencv.html

// https://github.com/opencv/opencv/blob/72c5ac37deaf59f73c918449a0f90e49f7866034/modules/calib3d/src/chessboard.cpp#L3634
// https://github.com/opencv/opencv/blob/72c5ac37deaf59f73c918449a0f90e49f7866034/modules/calib3d/src/calibinit.cpp#L486
// https://github.com/opencv/opencv/blob/72c5ac37deaf59f73c918449a0f90e49f7866034/modules/calib3d/src/calibinit.cpp#L2000
// https://github.com/opencv/opencv/blob/72c5ac37deaf59f73c918449a0f90e49f7866034/modules/calib3d/src/calibinit.cpp#L742

use image::{ImageBuffer, Luma, DynamicImage, Rgb};
use imageproc::{drawing};
use std::collections::HashMap;

use crate::common::get_pixel_coord;

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
    corner_distance: u32,
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

    // Get all 8 pixels intensity around the current corner (they may be further apart than 1 pixel, thus the corner_distance)

    // | i4 | i3 | i2 | 
    // | i5 | __ | i1 | 
    // | i6 | i7 | i8 |

    let delta_s = corner_distance as i32;
    let delta_0 = 0;

    let i1: f64 = image[get_pixel_coord((x_c + delta_s, y_c + delta_0), width, height)][0] as f64;
    let i2: f64 = image[get_pixel_coord((x_c + delta_s, y_c + delta_s), width, height)][0] as f64;
    let i3: f64 = image[get_pixel_coord((x_c + delta_0, y_c + delta_s), width, height)][0] as f64;
    let i4: f64 = image[get_pixel_coord((x_c - delta_s, y_c + delta_s), width, height)][0] as f64;
    let i5: f64 = image[get_pixel_coord((x_c - delta_s, y_c + delta_0), width, height)][0] as f64;
    let i6: f64 = image[get_pixel_coord((x_c - delta_s, y_c - delta_s), width, height)][0] as f64;
    let i7: f64 = image[get_pixel_coord((x_c + delta_0, y_c - delta_s), width, height)][0] as f64;
    let i8: f64 = image[get_pixel_coord((x_c + delta_s, y_c - delta_s), width, height)][0] as f64;

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

    let a = ((self_x_f64 - closer_neighbor_1_x), (self_y_f64 - closer_neighbor_1_y));
    let b = ((self_x_f64 - closer_neighbor_2_x), (self_y_f64 - closer_neighbor_2_y));

    let cos_theta = (a.0 * b.0 + a.1 * b.1) / (norm(a) * norm(b));

    //let theta = cos_theta.acos().to_degrees();

    if cos_theta < t_cosine_threshold {
        CornerFilterResult::RealCorner
    } else {
        // println!("t_cosine_threshold is {}", t_cosine_threshold);
        // println!("threshold is {}", t_cosine_threshold.acos().to_degrees());
        // println!("theta is {}", theta);
        CornerFilterResult::FakeCorner
    }
}

fn distance((a_x, a_y) : (f64, f64), (b_x, b_y) : (f64, f64)) -> f64 {
    ((a_x - b_x).powi(2) + (a_y - b_y).powi(2)).sqrt()
}

fn norm((a_x, a_y) : (f64, f64)) -> f64 {
    (a_x.powi(2) + a_y.powi(2)).sqrt()
}

pub struct ChessboardDetectorParameters {
    pub r: f64,
    pub p: f64,
    pub d: f64,
    pub t: f64,
    pub corner_distance: u32,
}

pub fn compute_adaptive_parameters(a_min: f64, a_max: f64) -> ChessboardDetectorParameters {

    let r = 0.7f64 * a_min;
    let p = 0.3f64 * a_max / a_min;
    let d = 2.0f64 * a_max;
    let t = 0.4f64 * a_max / a_min;
    let corner_distance = 5u32; // TODO : find good distance for this

    ChessboardDetectorParameters { r, p, d, t, corner_distance }
}

pub struct ClosestNeighborDistanceHistogram {
    histogram: HashMap::<u32, u32>, // key : distance, value: number of elements at that distance.
    number_of_values: u32,
    peak_index: u32,
}

impl ClosestNeighborDistanceHistogram {
    pub fn window_subset_ratio(&self, window_center: u32, window_size: u32) -> f64 {
        let mut nb_element_in_windows = 0;

        let window_min = window_center as i32 - window_size as i32;
        let window_max = window_center as i32 + window_size as i32;

        for (key, value) in &self.histogram {
            let key = *key as i32;
            if window_min <= key && key <= window_max {
                nb_element_in_windows += value;
            }
        }

        nb_element_in_windows as f64 / self.number_of_values as f64
    }

    pub fn get_peak_value(&self) -> u32 {
        self.histogram[&self.peak_index]
    }

    pub fn get_peak_index(&self) -> u32 {
        self.peak_index
    }

    pub fn window_size_that_cover_x_percent(&self, x: f64) -> u32 {
        assert!(x <= 1.0f64);
        assert!(x > 0.0f64);

        let step = (self.number_of_values as f64 / 100.0f64) as u32;

        let mut window_size = 0u32;
        let mut subset_ratio = 0f64;

        while subset_ratio < x {
            subset_ratio = self.window_subset_ratio(self.peak_index, window_size);
            window_size += step;
        }

        window_size
    }

    pub fn mean_val_and_std_dev_for_window(&self, window_size: u32) -> (f64, f64) {

        let mut sub_window_histogram = HashMap::<u32, u32>::new();
        let window_center = self.peak_index;

        let window_min = window_center as i32 - window_size as i32;
        let window_max = window_center as i32 + window_size as i32;

        let mut number_of_elem_in_window = 0;
        let mut sum = 0.0f64;

        for (key, value) in &self.histogram {
            let key = *key as i32;
            if window_min <= key && key <= window_max {
                sub_window_histogram.insert(key as u32, *value);
                number_of_elem_in_window += *value;
                sum += (key as f64) * (*value as f64);
            }
        }

        let mean = sum / number_of_elem_in_window as f64;

        let mut std_dev = 0.0f64;

        for (key, value) in &self.histogram {
            let key = *key as i32;
            if window_min <= key && key <= window_max {
                let value = *value as f64;
                let key = key as f64;
                let distance_to_mean = (key as f64 - mean).abs();
                std_dev += value * distance_to_mean * distance_to_mean;
            }
        }

        std_dev = std_dev / number_of_elem_in_window as f64;
        std_dev = std_dev.sqrt();

        (mean, std_dev)
    }
}

pub fn compute_closest_neighbor_distance_histogram(corners: &[CornerLocation]) -> ClosestNeighborDistanceHistogram {
    let mut histogram = HashMap::<u32, u32>::new();
    let mut sum = 0;

    for index_1 in 0..corners.len() {
        let (self_x, self_y) = corners[index_1];
        let self_x_f64 = self_x as f64;
        let self_y_f64 = self_y as f64;

        let mut closest_distance = std::f64::MAX;
        let mut has_run = false;

        for index_2 in (index_1 + 1)..corners.len() {

            // Don't make sense to check if pixel with itself
            if index_1 == index_2 {
                continue;
            }

            let (other_x, other_y) = corners[index_2];

            let other_x_f64 = other_x as f64;
            let other_y_f64 = other_y as f64;

            let distance = distance((self_x_f64, self_y_f64), (other_x_f64, other_y_f64));

            if distance <= closest_distance {
                closest_distance = distance;
                has_run = true;
            }
        }

        if has_run {
            let distance = closest_distance as u32;
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

    ClosestNeighborDistanceHistogram {
        histogram: histogram,
        number_of_values: sum,
        peak_index: peak_index,
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EliminationCause {
    Symmetry,
    Distance,
    Angle,
}

pub struct FilteringResult {
    pub remaining_corners: Vec<(i32, i32)>,
    pub filtered_out_corners: Vec<(i32, i32, EliminationCause)>,
}

pub fn filter_out_corners(
    chessboard_parameters: &ChessboardDetectorParameters, 
    corners: &Vec<(i32, i32)>, 
    blurred_gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
) -> FilteringResult {
    let mut output_corners = corners.clone();
    //let mut number_of_iterations = 0;
    let mut has_eliminated_at_least_a_point_this_loop_iteration = true;
    
    let mut wrong_corners_indexes: Vec<(usize, EliminationCause)> = vec!();
    let mut corners_eliminated : Vec<(i32, i32, EliminationCause)> = Vec::new();

    while has_eliminated_at_least_a_point_this_loop_iteration {

        //println!("iteration {} ===============", number_of_iterations);
        has_eliminated_at_least_a_point_this_loop_iteration = false;

        for (index, (x, y)) in output_corners.iter().enumerate()  {

            let corner_location = (*x, *y);

            let corner_filter = 
                apply_center_symmetry_filter(
                    chessboard_parameters.p,
                    chessboard_parameters.corner_distance,
                    &blurred_gray_image,
                    corner_location
                );

            if corner_filter == CornerFilterResult::FakeCorner {
                //println!("we have got a fake corner because of symmetry filter at {} {}", corner_location.0, corner_location.1);
                wrong_corners_indexes.push((index, EliminationCause::Symmetry));
                has_eliminated_at_least_a_point_this_loop_iteration = true;
                continue;
            }

            let corner_filter = apply_neighbor_distance_filter(
                chessboard_parameters.d,
                &output_corners,
                index,
            );

            if corner_filter == CornerFilterResult::FakeCorner {
                //println!("we have got a fake corner because of neighbor distance filter at {} {}", corner_location.0, corner_location.1);
                wrong_corners_indexes.push((index, EliminationCause::Distance));
                has_eliminated_at_least_a_point_this_loop_iteration = true;
                continue;
            }

            let corner_filter = apply_neighbor_angle_filter(
                chessboard_parameters.t,
                &output_corners,
                index,
            );

            if corner_filter == CornerFilterResult::FakeCorner {
                //println!("we have got a fake corner because of angle criterion at {} {}", corner_location.0, corner_location.1);
                wrong_corners_indexes.push((index, EliminationCause::Angle));
                has_eliminated_at_least_a_point_this_loop_iteration = true;
                continue;
            }
        }

        wrong_corners_indexes.reverse();

        for (index_to_remove, elimination_cause) in &wrong_corners_indexes {
            let corner = corners[*index_to_remove];
            corners_eliminated.push((corner.0, corner.1, *elimination_cause));
        }

        for (index_to_remove, _elimination_cause) in &wrong_corners_indexes {
            output_corners.remove(*index_to_remove);
        }

        wrong_corners_indexes.clear();

        //number_of_iterations += 1;
    }

    //println!("we have eliminated  {} corners in {} round(s)", corners_eliminated.len(), number_of_iterations);

    FilteringResult {
        remaining_corners: output_corners,
        filtered_out_corners: corners_eliminated
    }
}

pub fn draw_filtering_result(
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    draw_eliminated: bool,
    filtering_result: &FilteringResult
) {
    if draw_eliminated {
        for (x, y, elimination_cause) in &filtering_result.filtered_out_corners  {
            let color = match elimination_cause {
                EliminationCause::Distance => Rgb([255, 0, 0]),
                EliminationCause::Symmetry => Rgb([0, 255, 0]),
                EliminationCause::Angle => Rgb([0, 0, 255]),
            };

            drawing::draw_filled_circle_mut(
                canvas, 
                (*x as i32 , *y as i32),
                1i32,
                color);
        }
    } else {

        for (x, y) in &filtering_result.remaining_corners  {
            drawing::draw_filled_circle_mut(
                canvas, 
                (*x as i32 , *y as i32),
                1i32,
                Rgb([255, 0, 0]));
        }
    }

    let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    imgshow::imgshow(&out_img);
}
