use std::ops::{Deref, Range};

use image::{DynamicImage, ImageBuffer, Pixel, Rgba};
use map_in_place::MapVecInPlace;

pub trait FastImagePixel: Copy + Clone {}

impl<T> FastImagePixel for T where T: Copy + Clone {}

#[derive(Clone)]
pub struct FastImage<T> {
    data: Vec<T>,
    width_stride: usize,
    width: usize,
    height: usize,
}

impl<T> FastImage<T>
where
    T: FastImagePixel,
{
    pub fn new(width: usize, height: usize, fill: T) -> FastImage<T> {
        FastImage {
            data: vec![fill; height * width],
            width,
            height,
            width_stride: width,
        }
    }

    pub fn from_data(width: usize, height: usize, data: Vec<T>) -> FastImage<T> {
        FastImage {
            data,
            width,
            height,
            width_stride: width,
        }
    }

    /// Updates width and height of the image, but does not update its contents.
    fn maximize(&mut self, width: usize, height: usize, fill: &T) {
        debug_assert!(width >= self.width, height >= self.height);
        ((self.width * self.height)..(width * height)).for_each(|_| self.data.push(fill.clone()));

        self.width = width;
        self.height = height;
    }

    /// Updates width and height of the image, but does not update its contents.
    fn minimize(&mut self, width: usize, height: usize) {
        debug_assert!(width <= self.width, height <= self.height);
        self.width = width;
        self.height = height;
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub unsafe fn get_row_unchecked(&self, mut x: Range<usize>, y: usize) -> &[T] {
        debug_assert!(
            x.start < self.width && x.end <= self.width && y < self.height,
            "get_row_unchecked coordinates out of bounds {} {} {}",
            x.start,
            x.end,
            y
        );

        let rowoffs = y * self.width_stride;
        x.end += rowoffs;
        x.start += rowoffs;
        self.data.get_unchecked(x)
    }

    pub unsafe fn put_row_unchecked(&mut self, mut x: Range<usize>, y: usize, data: &[T]) {
        debug_assert!(
            x.start < self.width
                && x.end <= self.width
                && y < self.height
                && data.len() == x.end - x.start,
            "put_row_unchecked coordinates out of bounds"
        );

        let rowoffs = y * self.width_stride;
        x.end += rowoffs;
        x.start += rowoffs;
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            self.data.get_unchecked_mut(x).as_mut_ptr(),
            data.len(),
        );
    }

    pub unsafe fn get_pixel_unchecked(&self, x: usize, y: usize) -> &T {
        debug_assert!(
            x < self.width && y < self.height,
            "get_pixel_unchecked coordinates out of bounds"
        );

        self.data.get_unchecked(x + y * self.width_stride)
    }

    pub unsafe fn put_pixel_unchecked(&mut self, x: usize, y: usize, value: T) {
        debug_assert!(
            x < self.width && y < self.height,
            "put_pixel_unchecked coordinates out of bounds"
        );

        *self.data.get_unchecked_mut(x + y * self.width_stride) = value;
    }

    pub fn rotate90_mut(&mut self) {
        let mut temp = vec![self.data[0]; self.width * self.height];

        for row_idx in 0..self.height {
            unsafe {
                let row = self.get_row_unchecked(0..self.width, row_idx);

                for (i, count) in ((self.height - 1 - row_idx)..temp.len())
                    .step_by(self.height)
                    .zip(0..self.width)
                {
                    *temp.get_unchecked_mut(i) = *row.get_unchecked(count);
                }
            }
        }

        self.data = temp;
        std::mem::swap(&mut self.width, &mut self.height);
        self.width_stride = self.width;
    }

    pub fn rotate270_mut(&mut self) {
        let mut temp = vec![self.data[0]; self.width * self.height];

        for row_idx in 0..self.height {
            unsafe {
                let row = self.get_row_unchecked(0..self.width, row_idx);

                for (i, count) in (row_idx..temp.len())
                    .step_by(self.height)
                    .zip((0..self.width).rev())
                {
                    *temp.get_unchecked_mut(i) = *row.get_unchecked(count);
                }
            }
        }

        self.data = temp;
        std::mem::swap(&mut self.width, &mut self.height);
        self.width_stride = self.width;
    }

    pub fn from_image<P: 'static, C: Deref<Target = [P::Subpixel]>, CvFn>(
        image: &ImageBuffer<P, C>,
        conversion: CvFn,
    ) -> FastImage<T>
    where
        CvFn: Fn(&P) -> T,
        P: Pixel,
    {
        let w = image.width() as usize;

        FastImage {
            data: image.pixels().map(|p| conversion(&p)).collect(),
            width: w,
            height: image.height() as usize,
            width_stride: w,
        }
    }

    pub fn into_image<CvFn, P: 'static>(
        self,
        conversion: CvFn,
    ) -> ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>
    where
        CvFn: Fn(&T) -> P,
        P: Pixel,
    {
        let mut im: ImageBuffer<P, Vec<P::Subpixel>> =
            ImageBuffer::new(self.width as u32, self.height as u32);
        im.pixels_mut().enumerate().for_each(|(i, p)| {
            *p = conversion(unsafe { self.get_pixel_unchecked(i % self.width, i / self.width) })
        });
        im
    }
}

pub fn easy_resize(image: &DynamicImage, new_width: usize, new_height: usize) -> DynamicImage {
    DynamicImage::ImageRgba8(
        resize(
            &FastImage::from_image(&image.to_rgba(), |rgba| *rgba),
            new_width,
            new_height,
            &|left, right| {
                Rgba([
                    ((right.0[0] as u16 + left.0[0] as u16 + 1) / 2) as u8,
                    ((right.0[1] as u16 + left.0[1] as u16 + 1) / 2) as u8,
                    ((right.0[2] as u16 + left.0[2] as u16 + 1) / 2) as u8,
                    ((right.0[3] as u16 + left.0[3] as u16 + 1) / 2) as u8,
                ])
            },
            &Rgba([0, 0, 0, 0]),
            &|p| p.0[0] as u16 + p.0[1] as u16 + p.0[2] as u16,
            [-1, 0, 1, -2, 0, 2, -1, 0, 1],
        )
        .into_image(|rgba| *rgba),
    )
}

pub fn resize<T, I, LumaFn>(
    image: &FastImage<T>,
    new_width: usize,
    new_height: usize,
    interpolation_fn: &I,
    fill: &T,
    luma_fn: &LumaFn,
    kernel: [i16; 3 * 3],
) -> FastImage<T>
where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
    LumaFn: Fn(&T) -> u16,
{
    let (width, height) = image.dimensions();

    let mut image = image.clone();

    if new_width > width {
        image = maximize_seam(&image, new_width, interpolation_fn, fill, luma_fn, kernel);
    } else if new_width < width {
        image = minimize_seam(&image, new_width, luma_fn, kernel);
    }

    image.rotate90_mut();

    if new_height > height {
        image = maximize_seam(&image, new_height, interpolation_fn, fill, luma_fn, kernel);
    } else if new_height < height {
        image = minimize_seam(&image, new_height, luma_fn, kernel);
    }

    image.rotate270_mut();

    image
}

pub fn maximize_seam<T, I, LumaFn>(
    image: &FastImage<T>,
    new_width: usize,
    interpolation_fn: &I,
    fill: &T,
    luma_fn: &LumaFn,
    kernel: [i16; 3 * 3],
) -> FastImage<T>
where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
    LumaFn: Fn(&T) -> u16,
{
    let current_width = image.dimensions().0;

    let mut out_a = image.clone();
    let mut out_b = FastImage::new(out_a.width, out_a.height, out_a.data[0]);

    let mut swap = false;

    for _seam_idx in current_width..new_width {
        // calculate pixel energies
        let mut energies = energy(if swap { &out_b } else { &out_a }, luma_fn, kernel);
        // add up energies
        add_waterfall(&mut energies);
        // find seam path
        let seam_path = seam_path(&energies);
        // shift to seam
        if swap {
            shift_maximize(&out_b, &mut out_a, &seam_path, interpolation_fn, fill);
        } else {
            shift_maximize(&out_a, &mut out_b, &seam_path, interpolation_fn, fill);
        }

        swap = !swap;
    }

    if swap {
        out_b
    } else {
        out_a
    }
}

/// Splits the image top (0) to bottom (seam_path.len() - 1) along the seam
/// interpolation_fn is a function which should take the left and right pixel which have been split, and return a color for the seam
pub fn shift_maximize<I, T>(
    image: &FastImage<T>,
    output: &mut FastImage<T>,
    seam_path: &[usize],
    interpolation_fn: I,
    fill: &T,
) where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
{
    let (width, height) = image.dimensions();
    output.maximize(width + 1, height, fill);

    for (row_idx, seam_pos) in (0..height).into_iter().zip(seam_path.into_iter()) {
        unsafe {
            output.put_row_unchecked(
                0..*seam_pos,
                row_idx,
                image.get_row_unchecked(0..*seam_pos, row_idx),
            );

            output.put_row_unchecked(
                (*seam_pos + 1)..(width + 1),
                row_idx,
                image.get_row_unchecked(*seam_pos..width, row_idx),
            );

            if *seam_pos != 0 {
                let (left, right) = image
                    .get_row_unchecked((seam_pos - 1)..(seam_pos + 1).min(width), row_idx)
                    .split_at(1);

                output.put_pixel_unchecked(
                    *seam_pos,
                    row_idx,
                    interpolation_fn(&left.get_unchecked(0), &right.get_unchecked(0)),
                );
            } else {
                output.put_pixel_unchecked(
                    *seam_pos,
                    row_idx,
                    *image.get_pixel_unchecked(*seam_pos, row_idx),
                );
            }
        }
    }
}

pub fn minimize_seam<T, LumaFn>(
    image: &FastImage<T>,
    new_width: usize,
    luma_fn: &LumaFn,
    kernel: [i16; 3 * 3],
) -> FastImage<T>
where
    LumaFn: Fn(&T) -> u16,
    T: FastImagePixel,
{
    let current_width = image.dimensions().0;

    let mut out_a = image.clone();
    let mut out_b = FastImage::new(out_a.width, out_a.height, out_a.data[0]);

    let mut swap = false;

    for _seam_idx in (new_width..current_width).rev() {
        // calculate pixel energies
        let mut energies = energy(if swap { &out_b } else { &out_a }, luma_fn, kernel);
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
        out_b
    } else {
        out_a
    }
}

pub fn shift_minimize<T>(image: &FastImage<T>, output: &mut FastImage<T>, seam_path: Vec<usize>)
where
    T: FastImagePixel,
{
    let (width, height) = image.dimensions();
    output.minimize(width - 1, height);

    for (row_idx, seam_pos) in (0..height).into_iter().zip(seam_path.into_iter()) {
        unsafe {
            output.put_row_unchecked(
                0..seam_pos,
                row_idx,
                image.get_row_unchecked(0..seam_pos, row_idx),
            );

            output.put_row_unchecked(
                seam_pos..width - 1,
                row_idx,
                image.get_row_unchecked((seam_pos + 1)..width, row_idx),
            );
        }
    }
}

pub fn seam_path(image: &FastImage<u16>) -> Vec<usize> {
    let (width, height) = image.dimensions();
    let mut output: Vec<usize> = vec![0; height as usize];

    // find the least energetic pixel in the bottom row of pixels
    let mut least_energetic_idx = {
        let mut min = u16::max_value();
        let mut min_idx = 0;

        for x in 0..width {
            let val = unsafe { *image.get_pixel_unchecked(x, height - 1) };
            if val < min {
                min = val;
                min_idx = x;
            }
        }

        min_idx as usize
    };

    output[height as usize - 1] = least_energetic_idx;

    // find a seam from bottom to top
    for y in (0..(height - 1)).rev() {
        let mut min = u16::max_value();

        for scan_x in if least_energetic_idx != 0 {
            least_energetic_idx - 1
        } else {
            0
        }..=if least_energetic_idx < width - 1 {
            least_energetic_idx + 1
        } else {
            width - 1
        } {
            let val = unsafe { *image.get_pixel_unchecked(scan_x, y) };
            if val < min {
                min = val;
                least_energetic_idx = scan_x;
            }
        }

        output[y as usize] = least_energetic_idx;
    }

    output
}

pub fn add_waterfall(image: &mut FastImage<u16>) {
    let (width, height) = image.dimensions();

    for y in 1..height {
        //P1P2P3
        //  CP

        // left border
        unsafe {
            let left = *image.get_pixel_unchecked(0, y - 1);
            let right = *image.get_pixel_unchecked(1, y - 1);
            image.put_pixel_unchecked(0, y, *image.get_pixel_unchecked(0, y) + left.min(right));
        }

        for x in 1..width - 2 {
            unsafe {
                // find the pixel with the lowest value of the three pixels above
                image.put_pixel_unchecked(
                    x,
                    y,
                    image.get_pixel_unchecked(x, y)
                        + image.get_pixel_unchecked(x - 1, y - 1).min(
                            image
                                .get_pixel_unchecked(x, y - 1)
                                .min(image.get_pixel_unchecked(x + 1, y - 1)),
                        ),
                );
            }
        }

        // right border
        unsafe {
            let right = *image.get_pixel_unchecked(width - 1, y - 1);
            let left = *image.get_pixel_unchecked(width - 2, y - 1);
            image.put_pixel_unchecked(
                width - 1,
                y,
                *image.get_pixel_unchecked(width - 1, y) + left.min(right),
            );
        }
    }
}

pub fn energy<T, LumaFn>(
    image: &FastImage<T>,
    luma_fn: &LumaFn,
    kernel: [i16; 3 * 3],
) -> FastImage<u16>
where
    LumaFn: Fn(&T) -> u16,
    T: FastImagePixel,
{
    let (width, height) = image.dimensions();

    let mut image_luma: FastImage<u16> = FastImage::new(width, height, 0);
    let mut output: FastImage<i16> = FastImage::new(width, height, 0);

    let mut temp_buf = vec![0; width];
    for row_idx in 0..image_luma.height {
        unsafe {
            temp_buf
                .iter_mut()
                .zip(image.get_row_unchecked(0..width, row_idx).iter())
                .for_each(|(luma, rgb)| *luma = luma_fn(rgb));

            image_luma.put_row_unchecked(0..width, row_idx, &temp_buf);
        }
    }

    let kernel_offsets = [
        ((width as i32) + 1, -1, -1),
        (width as i32, 0, -1),
        ((width as i32) - 1, 1, -1),
        (1, -1, 0),
        (0, 0, 0),
        (-1, 1, 0),
        (-(width as i32) + 1, -1, 1),
        (-(width as i32), 0, 1),
        (-(width as i32) - 1, 1, 1),
    ];

    for ((kernel_offset, kx, ky), kernel) in
        kernel_offsets.iter().cloned().zip(kernel.iter().cloned())
    {
        for row_idx in (ky.max(0) as usize * width..(height as i32 + ky.min(0)) as usize * width)
            .step_by(width)
        {
            for x in kx.max(0) as usize..(width as i32 + kx.min(0)) as usize {
                unsafe {
                    *output
                        .data
                        .get_unchecked_mut(((row_idx + x) as i32 + kernel_offset) as usize) +=
                        *image_luma.data.get_unchecked(row_idx + x) as i16 * kernel;
                }
            }
        }
    }

    FastImage {
        width,
        height,
        width_stride: width,
        data: output.data.map_in_place(|int| int.abs() as u16),
    }
}
