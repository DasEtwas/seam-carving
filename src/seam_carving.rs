use std::{ops::Deref};

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Pixel, Rgb};

pub struct FastImage<T> {
    data: Vec<T>,
    width: u32,
    height: u32,
}

impl<T> FastImage<T> where T: Pixel + 'static {
    #[inline(always)]
    pub fn new(width: u32, height: u32, t: &T) -> FastImage<T> {
        FastImage {
            data: vec![*t; height as usize * width as usize],
            width,
            height,
        }
    }

    pub fn from_image<C: Deref<Target=[T::Subpixel]>>(image: &ImageBuffer<T, C>) -> FastImage<T> {
        let w = image.width();
        let h = image.height();

        FastImage {
            data: {
                let mut vec = Vec::with_capacity(w as usize * h as usize);
                for pixel in image.pixels() {
                    vec.push(pixel.clone());
                }
                vec
            },
            width: w,
            height: h,
        }
    }

    // doesn't actually resize anything
    fn minimize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    // doesn't actually resize anything
    fn maximize(&mut self, width: u32, height: u32, t: &T) {
        for _ in (self.width * self.height)..(width * height) {
            self.data.push(t.clone());
        }

        self.width = width;
        self.height = height;
    }

    pub fn into_image(self) -> ImageBuffer<T, Vec<<T as Pixel>::Subpixel>> {
        let mut im: ImageBuffer<T, Vec<<T as Pixel>::Subpixel>> =
            ImageBuffer::new(self.width, self.height);
        im.pixels_mut()
            .enumerate()
            .for_each(|(i, p)| *p = *self.get_pixel(i as u32 % self.width, i as u32 / self.width));
        im
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[inline(always)]
    pub fn get_pixel(&self, x: u32, y: u32) -> &T {
        &self.data[x as usize + y as usize * self.width as usize]
    }

    #[inline(always)]
    pub fn put_pixel(&mut self, x: u32, y: u32, value: T) {
        self.data[x as usize + y as usize * self.width as usize] = value;
    }
}

#[inline(always)]
pub fn resize(image: &DynamicImage, new_width: u32, new_height: u32) -> DynamicImage {
    let (width, height) = image.dimensions();

    let mut image = image.clone();

    if new_width > width {
        image = maximize_seam(&image, new_width);
    } else if new_width < width {
        image = minimize_seam(&image, new_width);
    }

    image = image.rotate90();

    if new_height > height {
        image = maximize_seam(&image, new_height);
    } else if new_height < height {
        image = minimize_seam(&image, new_height);
    }

    image.rotate270()
}

#[inline(always)]
pub fn maximize_seam(image: &DynamicImage, new_width: u32) -> DynamicImage {
    let current_width = image.dimensions().0;

    let mut out_a = FastImage::from_image(&image.to_rgb());
    let mut out_b = FastImage::new(out_a.width, out_a.height, out_a.get_pixel(0, 0));

    let mut swap = false;

    for _seam_idx in current_width..new_width {
        // calculate pixel energies
        let mut energies = energy(if swap { &out_b } else { &out_a });
        // add up energies
        add_waterfall(&mut energies);
        // find seam path
        let seam_path = seam_path(&energies);
        // shift to seam
        if swap {
            shift_maximize(&out_b, &mut out_a, seam_path);
        } else {
            shift_maximize(&out_a, &mut out_b, seam_path);
        }

        swap = !swap;
    }

    if swap {
        DynamicImage::ImageRgb8(out_b.into_image())
    } else {
        DynamicImage::ImageRgb8(out_a.into_image())
    }
}

#[inline(always)]
pub fn shift_maximize(image: &FastImage<Rgb<u8>>, output: &mut FastImage<Rgb<u8>>, seam_path: Vec<usize>) {
    let (width, height) = image.dimensions();
    output.maximize(width + 1, height, &Rgb { data: [0, 0, 0] });

    for y in 0..height {
        for x in 0..(width + 1) {
            let seam_pos = seam_path[y as usize] as u32;
            let pix = if x < seam_pos {
                *image.get_pixel(x, y)
            } else if x == seam_pos {
                // left becomes the average of left and right
                let mut left = *image.get_pixel((x as i32 - 1).max(0) as u32, y);
                let right = *image.get_pixel(x, y);
                left.data = [
                    // elements cant exceed u8::MAX_VALUE
                    ((right.data[0] as u16 + left.data[0] as u16) / 2) as u8,
                    ((right.data[1] as u16 + left.data[1] as u16) / 2) as u8,
                    ((right.data[2] as u16 + left.data[2] as u16) / 2) as u8,
                ];
                left
            } else {
                *image.get_pixel((x as i32 - 1).max(0) as u32, y)
            };
            output.put_pixel(x, y, pix);
        }
    }
}

#[inline(always)]
pub fn minimize_seam(image: &DynamicImage, new_width: u32) -> DynamicImage {
    let current_width = image.dimensions().0;

    let mut out_a = FastImage::from_image(&image.to_rgb());
    let mut out_b = FastImage::new(out_a.width, out_a.height, out_a.get_pixel(0, 0));

    let mut swap = false;

    for _seam_idx in (new_width..current_width).rev() {
        // calculate pixel energies
        let mut energies = energy(if swap { &out_b } else { &out_a });
        // add up energies
        add_waterfall(&mut energies);
        // find seam path
        let seam_path = seam_path(&energies);

        // shift to seam
        if swap {
            shift_minimize(&out_b, &mut out_a, seam_path);
        } else {
            shift_minimize(&out_a, &mut out_b, seam_path);
        }

        swap = !swap;
    }

    if swap {
        DynamicImage::ImageRgb8(out_b.into_image())
    } else {
        DynamicImage::ImageRgb8(out_a.into_image())
    }
}

#[inline(always)]
pub fn shift_minimize(image: &FastImage<Rgb<u8>>, output: &mut FastImage<Rgb<u8>>, seam_path: Vec<usize>) {
    let (width, height) = image.dimensions();
    output.minimize(width - 1, height);

    for y in 0..height {
        let in_y_offs = (y * width) as usize;
        let out_y_offs = (y * (width - 1)) as usize;

        let seam_pos = seam_path[y as usize];
        let mut x: usize = 0;

        while x < seam_pos {
            output.data[out_y_offs + x] = image.data[in_y_offs + x];
            x += 1;
        }

        while x < width as usize - 1 {
            output.data[out_y_offs + x] = image.data[in_y_offs + x + 1];
            x += 1;
        }
    }
}

#[inline(always)]
pub fn seam_path(image: &FastImage<Luma<u16>>) -> Vec<usize> {
    let (width, height) = image.dimensions();
    let mut output: Vec<usize> = vec![0; height as usize];

    // find the least energetic pixel in the bottom row of pixels
    let mut last_idx = {
        let mut min = u16::max_value();
        let mut min_idx = 0;

        for x in 0..width {
            let val = image.get_pixel(x, height - 1).data[0];
            if val < min {
                min = val;
                min_idx = x;
            }
        }

        min_idx as usize
    };

    output[height as usize - 1] = last_idx;

    // find a seam from bottom to top
    //for y in (0..(height - 1)).rev() {

    // while seems to be faster
    let mut y = height - 1;
    while y != 0 {
        y -= 1;

        let mut min = u16::max_value();

        for x in ((last_idx as i32 - 1).max(0))..=((last_idx as i32 + 1).min(width as i32 - 1)) {
            let val = image.get_pixel(x as u32, y).data[0];
            if val < min {
                min = val;
                last_idx = x as usize;
            }
        }

        output[y as usize] = last_idx;
    }

    output
}

#[inline(always)]
pub fn add_waterfall(image: &mut FastImage<Luma<u16>>) {
    let (width, height) = image.dimensions();

    for y in 1..height {
        for x in 0..width {
            let prev_value = image.get_pixel(x, y).data[0];

            //P1P2P3
            //  CP

            // find the pixel of the three pixels above (x,y), set lowest_value_above to the value of the pixel with the lowest value
            let mut lowest_value_above = u16::max_value();
            for scan_x in (x as i32 - 1).max(0)..=(x as i32 + 1).min(width as i32 - 1) {
                let p = image.get_pixel(scan_x as u32, y - 1).data[0];
                if p < lowest_value_above {
                    lowest_value_above = p;
                }
            }

            image.put_pixel(
                x,
                y,
                Luma {
                    //data: [*lowest_value_above.iter().last().unwrap() + prev_value],
                    data: [lowest_value_above + prev_value],
                },
            )
        }
    }
}

#[inline(always)]
pub fn energy(image: &FastImage<Rgb<u8>>) -> FastImage<Luma<u16>> {
    let (width, height) = image.dimensions();

    let mut output: FastImage<Luma<u16>> = FastImage::new(width, height, &Luma { data: [0] });

    // cache warming
    let mut sum = 0;
    image.data.iter().for_each(|x| sum += x.data[0]);

    for x in 0..width {
        for y in 0..height {
            // always positive
            let gradient_abs = {
                let middle = luma(image.get_pixel(x, y));

                let lefttop = if x > 0 && y > 0 {
                    luma(image.get_pixel(x - 1, y - 1))
                } else {
                    middle
                };

                let righttop = if x < width - 1 && y > 0 {
                    luma(image.get_pixel(x + 1, y - 1))
                } else {
                    middle
                };

                let left = if x > 0 {
                    luma(image.get_pixel(x - 1, y))
                } else {
                    middle
                };

                let right = if x < width - 1 {
                    luma(image.get_pixel(x + 1, y))
                } else {
                    middle
                };

                let leftbottom = if x > 0 && y < height - 1 {
                    luma(image.get_pixel(x - 1, y + 1))
                } else {
                    middle
                };

                let rightbottom = if x < width - 1 && y < height - 1 {
                    luma(image.get_pixel(x + 1, y + 1))
                } else {
                    middle
                };

                //LT  RT
                //LLPPRR
                //LB  RB

                //(left - right + top - bottom).abs() // + (middle - left).abs() + (middle - right).abs() + (middle - bottom).abs() + (middle - top).abs()
                let sobel_gradient = righttop + rightbottom + (right - left) * 2 - lefttop - leftbottom;
                if sobel_gradient < 0 {
                    -sobel_gradient as u16
                } else {
                    sobel_gradient as u16
                }
            };

            let pixel = Luma {
                data: [gradient_abs],
            };

            output.put_pixel(x, y, pixel);
        }
    }

    output
}

// uses i16 so code energy() doesn't need to cast it from u16
#[inline(always)]
fn luma(rgb: &Rgb<u8>) -> i16 {
    // real luma variant
    // (rgb.data[0] as f32 * 0.299 + rgb.data[1] as f32 * 0.587 + rgb.data[2] as f32 * 0.11) as i16
    rgb.data[0] as i16 + rgb.data[1] as i16 + rgb.data[2] as i16
}
