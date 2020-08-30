
// https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf

use image::{DynamicImage, Rgb, Luma, imageops::FilterType, ImageBuffer};
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
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>,

    //starting_point_coordinates: (i32, i32),
    //index_for_starting_point: i32

) {
    
    let starting_point_coordinates = corners_centers.mean; // TODO : use mean or medium

    let mut distances_to_starting_point = 
        distance_to_points(starting_point_coordinates, possible_corners);

    {
        let index_for_starting_point = 0usize;
        let starting_point = distances_to_starting_point[index_for_starting_point].1;
        distances_to_starting_point.remove(index_for_starting_point);

        let other_points = 
            distances_to_starting_point
            .into_iter()
            .map(|(_dist, point)| point)
            .collect::<Vec<(i32, i32)>>();

        run_try(starting_point, &other_points, gray_image, canvas);
    }
}

fn run_try(
    starting_point: (i32, i32),
    other_possibles_corners: &Vec<(i32, i32)>,
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>
) {

    let other_points_and_distances_to_starting_point = 
        distance_to_points(starting_point, other_possibles_corners);

    assert!(other_possibles_corners.len() >= 7);

    for i in 0..7 {
    //for i in 0..1 {
        let (_dist, point) = other_points_and_distances_to_starting_point[i];
        let dir = diff(point, starting_point);
        let dir_length = norm(dir);
        let new_length = 0.4f64 * dir_length;
        let scaled_dir = (dir.0 as f64 * new_length / dir_length, dir.1 as f64 * new_length / dir_length);

        let perpendicular_dir = (scaled_dir.1, -scaled_dir.0);

        let grey_value_1_coord_coord_f64 = (scaled_dir.0 + perpendicular_dir.0, scaled_dir.1 + perpendicular_dir.1);
        let grey_value_2_coord_coord_f64 = (scaled_dir.0 - perpendicular_dir.0, scaled_dir.1 - perpendicular_dir.1);

        let grey_value_1_coord = (starting_point.0 + grey_value_1_coord_coord_f64.0 as i32, starting_point.1 + grey_value_1_coord_coord_f64.1 as i32);
        let grey_value_2_coord = (starting_point.0 + grey_value_2_coord_coord_f64.0 as i32, starting_point.1 + grey_value_2_coord_coord_f64.1 as i32);

        // TODO : check coordinates are on screen

        let grey_value_1 = gray_image[(grey_value_1_coord.0 as u32, grey_value_1_coord.1 as u32)][0];
        let grey_value_2 = gray_image[(grey_value_2_coord.0 as u32, grey_value_2_coord.1 as u32)][0];

        let diff = (grey_value_1 as i16 - grey_value_2 as i16).abs();

        // TODO : fix threshold comparison

        println!("diff is {}", diff);

        if diff >= 100 {

            // drawing::draw_filled_circle_mut(
            //     canvas, 
            //     grey_value_1_coord,
            //     1i32,
            //     Rgb([0, 255, 0])
            // );

            // drawing::draw_filled_circle_mut(
            //     canvas, 
            //     grey_value_2_coord,
            //     1i32,
            //     Rgb([255, 255, 0])
            // );

            drawing::draw_filled_circle_mut(
                canvas, 
                point,
                1i32,
                Rgb([0, 255, 0])
            );
        }

    }

    // for i in 0..4 {
    //     let (_dist, point) = other_points_and_distances_to_starting_point[i];
    //     drawing::draw_filled_circle_mut(
    //         canvas, 
    //         (point.0, point.1),
    //         1i32,
    //         Rgb([0, 255, 0])
    //     );
    // }
   
    let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    imgshow::imgshow(&out_img);
}

fn distance_to_points(
    point: (i32, i32),
    other_points: &[(i32, i32)],
) -> Vec<(f64, (i32, i32))> {
    let mut distances_to_starting_point = vec!();

    for (current_x, current_y) in other_points {
        let current_corner = (*current_x, *current_y);
        let distance = distance(current_corner, point);
        distances_to_starting_point.push((distance, current_corner));
    }

    distances_to_starting_point.sort_by(|a, b| {
        let a_distance: f64 = a.0;
        let b_distance: f64 = b.0;

        a_distance.partial_cmp(&b_distance).unwrap()
    });

    distances_to_starting_point
}

fn distance((a_x, a_y) : (i32, i32), (b_x, b_y) : (i32, i32)) -> f64 {
    ((a_x as f64 - b_x as f64).powi(2) + (a_y as f64 - b_y as f64).powi(2)).sqrt()
}

fn diff((a_x, a_y) : (i32, i32), (b_x, b_y) : (i32, i32)) -> (i32, i32) {
    ((a_x - b_x), (a_y - b_y))
}

fn norm((a_x, a_y) : (i32, i32)) -> f64 {
    ((a_x as f64).powi(2) + (a_y as f64).powi(2)).sqrt()
}