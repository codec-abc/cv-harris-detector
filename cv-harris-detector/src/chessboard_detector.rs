// https://pdfs.semanticscholar.org/0d33/65f9ff573ee03776f042f938f0d447945ccd.pdf
// https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf
// https://www.researchgate.net/publication/228345254_Automatic_calibration_of_digital_cameras_using_planar_chess-board_patterns/link/0fcfd5134c9811b4b7000000/download

use image::{DynamicImage, Rgb, Luma, ImageBuffer};
use imageproc::{drawing};

use crate::common::get_pixel_coord;

type CornerLocation = (i32, i32);

pub struct CornersMeanAndMedium {
    pub mean: CornerLocation,
    pub medium: CornerLocation,
}

pub fn find_corners_mean_and_medium(
    possible_corners: &Vec<CornerLocation>,
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

// //     (chessboard_size_x, chessboard_size_x): CornerLocation,

pub fn run_chessboard_detection(
    possible_corners: &Vec<CornerLocation>,
    corners_centers: &CornersMeanAndMedium,
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>,
) {
    
    let starting_point_coordinates = corners_centers.mean; // TODO : use mean or medium

    let mut distances_to_starting_point = 
        distance_to_points(starting_point_coordinates, possible_corners);

    // TODO: if it doesn't work try we a few alternating starting points
    {
        let index_for_starting_point = 0usize;
        let starting_point = distances_to_starting_point[index_for_starting_point].1;

        //distances_to_starting_point.remove(index_for_starting_point);

        // let other_points = 
        //     distances_to_starting_point
        //     .into_iter()
        //     .map(|(_dist, point)| point)
        //     .collect::<Vec<CornerLocation>>();

        let mut connections : Vec<Connection> = vec!();
        let mut remaining_points_to_explore: Vec<CornerLocation> = vec!();

        remaining_points_to_explore.push(starting_point);

        run_try(
            &mut remaining_points_to_explore,
            &mut connections,
            &possible_corners, 
            gray_image, 
            canvas
        );
    }
}

struct Connection {
    start: CornerLocation,
    end: CornerLocation,
    angle: f64,
    length: f64,
}

fn run_try(
    //starting_point: CornerLocation,
    remaining_points_to_explore: &mut Vec<CornerLocation>,
    connections: &mut Vec<Connection>,
    corners: &Vec<CornerLocation>,
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>
) {

    let width = gray_image.width();
    let height = gray_image.height();

    let mut explored_corners = vec!();


    while remaining_points_to_explore.len() > 0 {
        
        let starting_point = remaining_points_to_explore.remove(0);
        explored_corners.push(starting_point);

        let other_possibles_corners: Vec<CornerLocation> = 
            corners
            .clone()
            .into_iter()
            .filter(|corner| {
                !(corner.0 == starting_point.0 && corner.1 == starting_point.1) &&
                explored_corners.iter().find(|explored_corner| {
                    corner.0 == explored_corner.0 && corner.1 == explored_corner.1
                }).is_none()
            })
            .collect();


        let other_points_and_distances_to_starting_point = 
            distance_to_points(starting_point, &other_possibles_corners);

        let other_corners_count = std::cmp::min(7, other_possibles_corners.len());
        let mut main_directions = vec!();

        for i in 0..other_corners_count {
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

            let grey_value_1 = gray_image[get_pixel_coord((grey_value_1_coord.0 as i32, grey_value_1_coord.1 as i32), width, height)][0];
            let grey_value_2 = gray_image[get_pixel_coord((grey_value_2_coord.0 as i32, grey_value_2_coord.1 as i32), width, height)][0];

            let diff = grey_value_1 as i16 - grey_value_2 as i16;

            // TODO : fix threshold comparison or use sobel edge transform
            if diff.abs() >= 100 {
                main_directions.push(dir);
            }

        }

        if main_directions.len() >= 1 {

            let max = main_directions
                .iter()
                .max_by(|a, b|  {
                    let norm_a = norm((a.0, a.1));
                    let norm_b = norm((b.0, b.1));
                    norm_a.partial_cmp(&norm_b).unwrap()
                }).unwrap();

            let norm_max = norm((max.0, max.1));
            let right_point = (starting_point.0 + norm_max as i32, starting_point.1);
            let a = ((starting_point.0 - right_point.0), (starting_point.1 - right_point.1));

            let new_connections: Vec<(f64, f64, CornerLocation)> = 
                main_directions.iter().map(|dir| {
                let b = dir;

                let theta = 
                    (a.0 as f64 * b.1 as f64 - a.1 as f64 * b.0 as f64 )
                    .atan2(a.0 as f64 * b.0 as f64 + a.1 as f64 * b.1 as f64)
                    .to_degrees();
                
                let length = distance(a, *b);

                (theta, length, (starting_point.0 + dir.0, starting_point.1 + dir.1))
            }).collect();

            for (angle, length, end) in &new_connections {

                connections.push(
                    Connection {
                        start: starting_point,
                        end: *end,
                        angle: *angle,
                        length: *length
                    }
                );

                if 
                    explored_corners.iter().find(|explored_corner| 
                    {
                        explored_corner.0 == end.0 && explored_corner.1 == end.1
                    }).is_none() 
                    && 
                    remaining_points_to_explore.iter().find(|remaining_point_to_explore| {
                        remaining_point_to_explore.0 == end.0 && remaining_point_to_explore.1 == end.1
                    }).is_none()
                {
                    remaining_points_to_explore.push(*end);
                }
                
            }
        }
    }

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

        // drawing::draw_filled_circle_mut(
        //     canvas, 
        //     point,
        //     1i32,
        //     Rgb([0, 255, 0])
        // );
    
    for connection in connections {
        //let (_dist, point) = other_points_and_distances_to_starting_point[i];
        drawing::draw_filled_circle_mut(
            canvas, 
            connection.start,
            1i32,
            Rgb([0, 255, 0])
        );
    }
   
    println!("done");
    let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    imgshow::imgshow(&out_img);
}

fn distance_to_points(
    point: CornerLocation,
    other_points: &[CornerLocation],
) -> Vec<(f64, CornerLocation)> {
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

fn distance((a_x, a_y) : CornerLocation, (b_x, b_y) : CornerLocation) -> f64 {
    ((a_x as f64 - b_x as f64).powi(2) + (a_y as f64 - b_y as f64).powi(2)).sqrt()
}

fn diff((a_x, a_y) : CornerLocation, (b_x, b_y) : CornerLocation) -> CornerLocation {
    ((a_x - b_x), (a_y - b_y))
}

fn norm((a_x, a_y) : CornerLocation) -> f64 {
    ((a_x as f64).powi(2) + (a_y as f64).powi(2)).sqrt()
}