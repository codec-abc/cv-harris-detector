
// https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf

use image::{DynamicImage, Rgb, imageops::FilterType, ImageBuffer};
use imageproc::{drawing, filter};

pub struct CornersMeanAndMedium {
    pub mean: (i32, i32),
    pub medium: (i32, i32),
}

pub fn find_corners_mean_and_medium(
    possible_corners: &Vec<(i32, i32)>,
) -> CornersMeanAndMedium {
    let number_of_possibles_corners = possible_corners.len();
    assert!(number_of_possibles_corners > 0);

    let (mut mean_x, mut mean_y) = (0, 0);
    
    let (mut min_x, mut max_x)  = (std::i32::MAX, std::i32::MIN);
    let (mut min_y, mut max_y)  = (std::i32::MAX, std::i32::MIN);

    for corner in possible_corners {
        let x = corner.0;
        let y = corner.1;

        mean_x += x;
        mean_y += y;

        min_x = std::cmp::min(min_x, x);
        max_x = std::cmp::max(max_x, x);

        min_y = std::cmp::min(min_y, y);
        max_y = std::cmp::max(max_y, y);
    }

    mean_x = mean_x / number_of_possibles_corners as i32;
    mean_y = mean_y / number_of_possibles_corners as i32;

    let (medium_x, medium_y)  = ((min_x + max_x) / 2, (min_y + max_y) / 2);

    CornersMeanAndMedium {
        mean: (mean_x, mean_y),
        medium: (medium_x, medium_y),
    }
}

// //     (chessboard_size_x, chessboard_size_x): (i32, i32),

pub fn run_chessboard_detection(
    possible_corners: &Vec<(i32, i32)>,
    corners_centers: &CornersMeanAndMedium,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>,

    //starting_point_coordinates: (i32, i32),
    //index_for_starting_point: i32

) {
    let starting_point_coordinates = corners_centers.medium;
    let mut distances_to_starting_point = vec!();
    let index_for_starting_point = 0;

    let starting_point = (starting_point_coordinates.0 as f64, starting_point_coordinates.1 as f64);
    for (current_x, current_y) in possible_corners {
        let current_corner = (*current_x as f64, *current_y as f64);
        let distance = distance(current_corner, starting_point);
        distances_to_starting_point.push((distance, current_corner));
    }

    distances_to_starting_point.sort_by(|a, b| {
        let a_distance: f64 = a.0;
        let b_distance: f64 = b.0;

        a_distance.partial_cmp(&b_distance).unwrap()
    });

    let real_starting_point = distances_to_starting_point[index_for_starting_point as usize].1;

    drawing::draw_filled_circle_mut(
        canvas, 
        (real_starting_point.0 as i32, real_starting_point.1 as i32),
        1i32,
        Rgb([0, 255, 0])
    );

    let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    imgshow::imgshow(&out_img);
}

fn distance((a_x, a_y) : (f64, f64), (b_x, b_y) : (f64, f64)) -> f64 {
    ((a_x - b_x).powi(2) + (a_y - b_y).powi(2)).sqrt()
}