// clamp pixel coordinates. It should be better do to something like OpenCV does with BorderTypes
// example: https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_BorderTypes.html
pub fn get_pixel_coord((x, y): (i32, i32), width: u32, height: u32) -> (u32, u32) {
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