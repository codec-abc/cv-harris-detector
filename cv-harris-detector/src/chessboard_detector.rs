// https://pdfs.semanticscholar.org/0d33/65f9ff573ee03776f042f938f0d447945ccd.pdf
// https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/04.pdf
// https://www.researchgate.net/publication/228345254_Automatic_calibration_of_digital_cameras_using_planar_chess-board_patterns/link/0fcfd5134c9811b4b7000000/download

use bracket_color::prelude::HSV;
use image::{DynamicImage, Rgb, Luma, ImageBuffer};
use imageproc::{drawing};
//use rand::Rng;

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

// (chessboard_size_x, chessboard_size_x): CornerLocation,

pub fn run_chessboard_detection(
    possible_corners: &Vec<CornerLocation>,
    corners_centers: &CornersMeanAndMedium,
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>,
) {
    
    let current_point_coordinates = corners_centers.mean; // TODO : use mean or medium

    let distances_to_current_point = 
        distance_to_points(current_point_coordinates, possible_corners);

    // TODO: if it doesn't work try we a few alternating starting points
    {
        let index_for_current_point = 0usize;
        let current_point = distances_to_current_point[index_for_current_point].1;

        let mut connections : Vec<Connection> = vec!();
        let mut remaining_points_to_explore: Vec<CornerLocation> = vec!();

        remaining_points_to_explore.push(current_point);

        run_try(
            &mut remaining_points_to_explore,
            &mut connections,
            &possible_corners, 
            gray_image, 
            canvas
        );
    }
}

#[derive(Debug, Clone)]
struct Connection {
    start: CornerLocation,
    end: CornerLocation,
    angle: f64,
    length: f64,
}

fn run_try(
    remaining_points_to_explore: &mut Vec<CornerLocation>,
    connections: &mut Vec<Connection>,
    corners: &Vec<CornerLocation>,
    gray_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    canvas: &mut drawing::Blend<ImageBuffer<Rgb<u8>, Vec<u8>>>
) {

    let width = gray_image.width();
    let height = gray_image.height();

    let mut explored_corners = vec!();

    let mut nb_iter = 0;

    let mut discarded_too_long_edges = vec!();
    let mut discarded_edges_constrast_too_low = vec!();

    while remaining_points_to_explore.len() > 0 {
        // println!("remaining_points_to_explore {}", remaining_points_to_explore.len());
        nb_iter = nb_iter + 1;
        
        let current_point = remaining_points_to_explore.remove(0);
        explored_corners.push(current_point);

        let other_possibles_corners: Vec<CornerLocation> = 
            corners
            .clone()
            .into_iter()
            .filter(|corner| {
                !equals(*corner, current_point) &&
                explored_corners.iter().find(|explored_corner| {
                    equals(*corner, **explored_corner)
                }).is_none()
            })
            .collect();


        let mut other_points_and_distances_to_current_point = 
            distance_to_points(current_point, &other_possibles_corners);

        other_points_and_distances_to_current_point
            .sort_by(|a, b|  {
                a.0.partial_cmp(&b.0).unwrap()
            }
        );

        let other_corners_count = std::cmp::min(7, other_possibles_corners.len());

        let right_point = (current_point.0 + width as i32, current_point.1);
        let a = ((current_point.0 - right_point.0), (current_point.1 - right_point.1));
        let mut added_connections = 0;

        for i in 0..other_corners_count {
        
            let connections_cloned = connections.clone();
            let connections_to_current_point : Vec<_> = connections_cloned.iter().filter(|connec| {
                connec.start == current_point || connec.end == current_point 
            }).collect();

            println!("nb connetions to current point {}", connections_to_current_point.len());

            if connections_to_current_point.len() >= 4 {
                println!("Too many connections. stopping here for this point");
                break;
            }

            let (_dist, neighbor_point) = other_points_and_distances_to_current_point[i];
            let dir = diff(neighbor_point, current_point);
            let length_ratio = 0.5f64;
            
            let scaled_dir = (
                dir.0 as f64 * length_ratio, 
                dir.1 as f64 * length_ratio);

            let perpendicular_dir = (scaled_dir.1, -scaled_dir.0);

            let perpendicular_dir_factor = 0.4;

            let perpendicular_dir_scaled = 
                (
                    perpendicular_dir.0 * perpendicular_dir_factor, 
                    perpendicular_dir.1 * perpendicular_dir_factor
                );

            let grey_value_1_coord_f64 = (
                scaled_dir.0 + perpendicular_dir_scaled.0, 
                scaled_dir.1 + perpendicular_dir_scaled.1);

            let grey_value_2_coord_f64 = (
                scaled_dir.0 - perpendicular_dir_scaled.0, 
                scaled_dir.1 - perpendicular_dir_scaled.1);

            let grey_value_1_coord = (
                current_point.0 + grey_value_1_coord_f64.0 as i32, 
                current_point.1 + grey_value_1_coord_f64.1 as i32);

            let grey_value_2_coord = (
                current_point.0 + grey_value_2_coord_f64.0 as i32, 
                current_point.1 + grey_value_2_coord_f64.1 as i32);

            // TODO : check coordinates are on screen
            let grey_value_1 = gray_image[
                get_pixel_coord(
                    (grey_value_1_coord.0 as i32, grey_value_1_coord.1 as i32), 
                    width, 
                    height
                )][0];

            let grey_value_2 = gray_image[
                get_pixel_coord(
                    (grey_value_2_coord.0 as i32, grey_value_2_coord.1 as i32), 
                    width, 
                    height
                )][0];

            let diff = grey_value_1 as i16 - grey_value_2 as i16;

            // TODO : fix threshold comparison or use sobel edge transform
            if diff.abs() >= 100 {
                
                let b = dir;

                let angle = 
                    (a.0 as f64 * b.1 as f64 - a.1 as f64 * b.0 as f64 )
                    .atan2(a.0 as f64 * b.0 as f64 + a.1 as f64 * b.1 as f64)
                    .to_degrees();
                
                let length = distance(a, b);

                let mut add_point = true;

                let (count, total_distance) = connections_to_current_point.iter().fold(
                    (0, 0.0f64), |(count, value), connec| {
                        (count + 1, value + distance(connec.start, connec.end))
                    }
                );

                if count > 0
                {
                    let bound_margin = 0.5f64;
                    let absolute_margin = 20.0f64;

                    let average_distance = total_distance / (count as f64) + absolute_margin;
                    let new_distance = distance(current_point, neighbor_point) + absolute_margin;

                    let lower_bound = (1.0f64 - bound_margin) * average_distance;
                    let upper_bound = (1.0f64 + bound_margin) * average_distance;

                    if lower_bound <= new_distance && new_distance <= upper_bound {

                        println!("adding edge");
                        connections.push(
                            Connection {
                                start: current_point,
                                end: neighbor_point,
                                angle: angle,
                                length: length
                            }
                        );

                        added_connections = added_connections + 1;
                    } else {
                        add_point = false;

                        println!(
                            "skipping point {} {} because neighbor(s) distance is too big or too small. New distance: {}, neighbor average distance: {}", 
                            current_point.0,
                            current_point.1,
                            new_distance, 
                            average_distance
                        );

                        discarded_too_long_edges.push((current_point, neighbor_point, ));
                    }
                }
                else {
                    println!("point has no connection yet.");

                    connections.push(
                        Connection {
                            start: current_point,
                            end: neighbor_point,
                            angle: angle,
                            length: length
                        }
                    );

                    added_connections = added_connections + 1;
                }
                

                if add_point && 
                    //explored_corners.iter().find(|ex| equals(**ex, point)).is_none() && 
                    remaining_points_to_explore.iter().find(|r| equals(**r, neighbor_point)).is_none()
                {
                    remaining_points_to_explore.push(neighbor_point);
                }
            } else {
                discarded_edges_constrast_too_low.push(
                    (current_point, neighbor_point, grey_value_1_coord, grey_value_2_coord, diff));
            }

        }
    }

    println!("try done in {} steps", nb_iter);
    //let mut rng = rand::thread_rng();
    let nb_connections = connections.len();
    
    // for (index, connection) in connections.iter().enumerate() {

    //     let h = index as f32 / nb_connections as f32;
    //     let s = 0.5f32;
    //     let v = 1.0f32;

    //     let hsv = HSV::from_f32(h, s, v);

    //     let rgb = hsv.to_rgb();

    //     let r = (rgb.r * 255.0f32) as u8;
    //     let g = (rgb.g * 255.0f32) as u8;
    //     let b = (rgb.b * 255.0f32) as u8;

    //     drawing::draw_line_segment_mut(
    //         canvas, 
    //         (connection.start.0 as f32, connection.start.1 as f32), 
    //         (connection.end.0 as f32, connection.end.1 as f32), 
    //         Rgb([r, g, b])
    //         //Rgb([0, 255, 0])
    //     );
    // }

    // for removed_point in removed_points {
    //     drawing::draw_filled_circle_mut(
    //         canvas, 
    //         removed_point,
    //         1i32,
    //         Rgb([0, 255, 0])
    //     );
    // }

    // for (index, (start, end)) in discarded_too_long_edges.iter().enumerate() {
    //     let h = index as f32 / nb_connections as f32;
    //     let s = 0.5f32;
    //     let v = 1.0f32;

    //     let hsv = HSV::from_f32(h, s, v);

    //     let rgb = hsv.to_rgb();

    //     let r = (rgb.r * 255.0f32) as u8;
    //     let g = (rgb.g * 255.0f32) as u8;
    //     let b = (rgb.b * 255.0f32) as u8;

    //     drawing::draw_line_segment_mut(
    //         canvas, 
    //         (start.0 as f32, start.1 as f32), 
    //         (end.0 as f32, end.1 as f32), 
    //         Rgb([r, g, b])
    //     );
    // }

    for (index, (start, end, pixel1, pixel2, diff)) in discarded_edges_constrast_too_low.iter().enumerate() {

        if 0 <= index && index <= 3 {
            let h = index as f32 / 6 as f32;
            let s = 0.5f32;
            let v = 1.0f32;

            let hsv = HSV::from_f32(h, s, v);

            let rgb = hsv.to_rgb();

            let r = (rgb.r * 255.0f32) as u8;
            let g = (rgb.g * 255.0f32) as u8;
            let b = (rgb.b * 255.0f32) as u8;

            drawing::draw_line_segment_mut(
                canvas, 
                (start.0 as f32, start.1 as f32), 
                (end.0 as f32, end.1 as f32), 
                Rgb([r, g, b])
            );

            drawing::draw_filled_circle_mut(
                canvas, 
                *pixel1,
                1i32,
                Rgb([r, g, b])
            );

            drawing::draw_filled_circle_mut(
                canvas, 
                *pixel2,
                1i32,
                Rgb([r, g, b])
            );

            println!("diff is {}", diff);
        }
    }
   
    println!("done");
    let out_img = DynamicImage::ImageRgb8(canvas.0.clone());
    imgshow::imgshow(&out_img);
}

fn distance_to_points(
    point: CornerLocation,
    other_points: &[CornerLocation],
) -> Vec<(f64, CornerLocation)> {
    let mut distances_to_current_point = vec!();

    for (current_x, current_y) in other_points {
        let current_corner = (*current_x, *current_y);
        let distance = distance(current_corner, point);
        distances_to_current_point.push((distance, current_corner));
    }

    distances_to_current_point.sort_by(|a, b| {
        let a_distance: f64 = a.0;
        let b_distance: f64 = b.0;

        a_distance.partial_cmp(&b_distance).unwrap()
    });

    distances_to_current_point
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

fn equals((a_x, a_y) : CornerLocation, (b_x, b_y) : CornerLocation) -> bool {
    a_x == b_x && a_y == b_y
}