#[cfg(feature = "imagers")]
use image::{DynamicImage, ImageBuffer, Pixel, Rgba};
use std::ops::{Bound, Deref, Range, RangeBounds};

pub trait FastImagePixel: Copy + Clone {}

impl<T> FastImagePixel for T where T: Copy + Clone {}

/// Bitmap structure optimized for speed of access.
#[derive(Clone)]
pub struct FastImage<T> {
    data: Vec<T>,
    width_stride: usize,
    width: usize,
    height: usize,
}

impl<T> FastImage<T>
where
    T: FastImagePixel + Default,
{
    pub fn new_default(width: usize, height: usize) -> FastImage<T> {
        FastImage {
            data: vec![Default::default(); height * width],
            width,
            height,
            width_stride: width,
        }
    }
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

    pub fn from_data(width: usize, height: usize, data: Vec<T>) -> Option<FastImage<T>> {
        if data.len() >= width * height {
            Some(FastImage {
                data,
                width,
                height,
                width_stride: width,
            })
        } else {
            None
        }
    }

    /// Updates width and height of the image and fills the newly added empty space with `fill`.
    fn maximize(&mut self, width: usize, height: usize, fill: &T) {
        assert!(width >= self.width, height >= self.height);

        self.data = self
            .data
            .chunks_exact(self.width_stride)
            .flat_map(|row| {
                row.iter()
                    .cloned()
                    .chain(std::iter::repeat(*fill).take(width.saturating_sub(self.width_stride)))
            })
            .chain(std::iter::repeat(*fill).take(height.saturating_sub(self.height) * width))
            .collect();

        self.width_stride = width;
        self.width = width;
        self.height = height;
    }

    /// Updates width and height of the image and fills the newly added empty space with `fill`.
    ///
    /// If this image was previously not wider than set with this function, it will be skewed to avoid rebuilding each row.
    fn maximize_fast(&mut self, width: usize, height: usize, fill: &T) {
        assert!(width >= self.width, height >= self.height);

        self.data
            .extend(std::iter::repeat(fill).take((width * height).saturating_sub(self.data.len())));

        self.width = width;
        self.width_stride = self.width_stride.max(self.width);
        self.height = height;
    }

    /// Updates width and height of the image, but does not update its contents. (Cropping)
    fn minimize(&mut self, width: usize, height: usize) {
        assert!(width <= self.width, height <= self.height);

        self.width = width;
        self.height = height;
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn set_row(&mut self, x: impl RangeBounds<usize>, row: usize, data: &[T]) -> Option<()> {
        let rowoffs = row * self.width_stride;

        // exclusive range end
        let start = match x.end_bound() {
            Bound::Excluded(i) => *i,
            Bound::Included(i) => *i + 1,
            Bound::Unbounded => self.width,
        };
        // inclusive range start
        let end = match x.start_bound() {
            Bound::Excluded(i) => *i + 1,
            Bound::Included(i) => *i,
            Bound::Unbounded => 0,
        };

        if start >= self.width || end >= self.width {
            None
        } else {
            Some(unsafe {
                self.data
                    .get_unchecked_mut(rowoffs + start..rowoffs + end)
                    .copy_from_slice(data)
            })
        }
    }

    pub fn get_row(&self, x: impl RangeBounds<usize>, row: usize) -> Option<&[T]> {
        let rowoffs = row * self.width_stride;

        // exclusive range end
        let start = match x.end_bound() {
            Bound::Excluded(i) => *i,
            Bound::Included(i) => *i + 1,
            Bound::Unbounded => self.width,
        };
        // inclusive range start
        let end = match x.start_bound() {
            Bound::Excluded(i) => *i + 1,
            Bound::Included(i) => *i,
            Bound::Unbounded => 0,
        };

        if start >= self.width || end >= self.width {
            None
        } else {
            Some(unsafe { self.data.get_unchecked(rowoffs + start..rowoffs + end) })
        }
    }

    unsafe fn get_row_unchecked_mut(&mut self, mut x: Range<usize>, row: usize) -> &mut [T] {
        #[cfg(feature = "fastimage_debug")]
        assert!(
            x.start <= self.width && x.end <= self.width && row < self.height,
            "get_row_unchecked_mut coordinates out of bounds (start {}, end {}, y {})",
            x.start,
            x.end,
            row
        );

        let rowoffs = row * self.width_stride;
        x.end += rowoffs;
        x.start += rowoffs;
        self.data.get_unchecked_mut(x)
    }

    pub unsafe fn get_row_unchecked(&self, mut x: Range<usize>, row: usize) -> &[T] {
        #[cfg(feature = "fastimage_debug")]
        assert!(
            x.start <= self.width && x.end <= self.width && row < self.height,
            "get_row_unchecked coordinates out of bounds (start {}, end {}, y {})",
            x.start,
            x.end,
            row
        );

        let rowoffs = row * self.width_stride;
        x.end += rowoffs;
        x.start += rowoffs;
        self.data.get_unchecked(x)
    }

    pub unsafe fn set_row_unchecked(&mut self, mut x: Range<usize>, row: usize, data: &[T]) {
        #[cfg(feature = "fastimage_debug")]
        assert!(
            x.start <= self.width
                && x.end <= self.width
                && row < self.height
                && data.len() == x.end - x.start,
            "put_row_unchecked coordinates out of bounds (start {}, end {}, y {})",
            x.start,
            x.end,
            row
        );

        let rowoffs = row * self.width_stride;
        x.end += rowoffs;
        x.start += rowoffs;
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            self.data.get_unchecked_mut(x).as_mut_ptr(),
            data.len(),
        );
    }

    pub unsafe fn get_pixel_unchecked(&self, x: usize, y: usize) -> &T {
        #[cfg(feature = "fastimage_debug")]
        assert!(
            x < self.width && y < self.height,
            "get_pixel_unchecked coordinates out of bounds"
        );

        self.data.get_unchecked(x + y * self.width_stride)
    }

    pub unsafe fn set_pixel_unchecked(&mut self, x: usize, y: usize, value: T) {
        #[cfg(feature = "fastimage_debug")]
        assert!(
            x < self.width && y < self.height,
            "put_pixel_unchecked coordinates out of bounds"
        );

        *self.data.get_unchecked_mut(x + y * self.width_stride) = value;
    }

    /// Rotates this image clockwise 90 by degrees.
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

    /// Rotates this image counter-clockwise 90 by degrees.
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

    pub fn copy_from_image(&mut self, image: &FastImage<T>) {
        assert!(!image.data.is_empty(), "copying from empty image");

        self.data.extend(
            std::iter::repeat(image.data[0]).take(image.data.len().saturating_sub(self.data.len())),
        );
        self.data[..image.data.len()].copy_from_slice(&image.data);
        self.width_stride = image.width_stride;
        self.width = image.width;
        self.height = image.height;
    }

    /// Creates a `FastImage` from an `image::ImageBuffer`. The `conversion` function is used to map an image's pixels to different values for the `FastImage`.
    #[cfg(feature = "imagers")]
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

    #[cfg(feature = "imagers")]
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

/// Easy to use function to take a dynamic image, convert it to `Rgba`, and seam carve it.
///
/// Energy is calculated using the horizontal [Sobel] operator on pixel Luma.
///
/// [Sobel]: https://en.wikipedia.org/wiki/Sobel_operator#Technical_details
///
/// The minimum accepted image size is 3x1 (see `add_waterfall`)
#[cfg(feature = "imagers")]
pub fn easy_resize(image: &DynamicImage, new_width: usize, new_height: usize) -> DynamicImage {
    let mut image = FastImage::from_image(&image.to_rgba(), |rgba| *rgba);

    resize(
        &mut image,
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
        &|p| (p.0[0] as u32 + p.0[1] as u32 + p.0[2] as u32) * p.0[3] as u32,
        &[
            /*
            // Sobel
            [-1, 0, 1, -2, 0, 2, -1, 0, 1],
            [1, 2, 1, 0, 0, 0, -1, -2, -1],
            */
            /*
            // 8-bit Scharr
            [47, 0, -47, 162, 0, -162, 47, 0, 47],
            [47, 162, 47, 0, 0, 0, -47, -162, -47],
            */
            // Sobel-Feldman
            [-3, 0, 3, -10, 0, 10, -3, 0, 3],
            [3, 10, 3, 0, 0, 0, -3, -10, -3],
            /*
            // Prewitt
            [1, 0, -1, 1, 0, -1, 1, 0, -1],
            [1, 1, 1, 0, 0, 0, -1, -1, -1],
            */
            /*
            // idk
            [0, 1, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 1, 0, -1, 0, 0, 0],
            */
        ],
        &Rgba([0, 0, 0, 0]),
    );

    DynamicImage::ImageRgba8(image.into_image(|rgba| *rgba))
}

/// Resizes an image using the seam carving algorithm.
///
/// Resizing is done in two steps:
///  1. Horizontal resizing
///  2. Vertical resizing
///
/// When upscaling, every time a seam is created, it must be filled with a new pixel value.
/// For this, one must supply an `interpolation_fn`. For normal usage, a function with averages the two pixel values is recommended.
/// During upscaling, temporary image buffers are initialized with `fill`.
///
/// For every pixel of resizing, a seam is found along the path of least energy. Energy is calculated using the supplied 3x3 `kernel` which is applied onto the luma representation of the `image`, calculated with the conversion function `luma_fn`.
///
/// Due to the waterfall adding step, the luma values (u16) can get quite large, which is why the luma function is recommended to output values considerably smaller than u16::MAX (e.g. 0..512). Large luma values will saturate the waterfall image pixel values and decrease seam-finding quality.
pub fn resize<T, I, LumaFn>(
    image: &mut FastImage<T>,
    new_width: usize,
    new_height: usize,
    interpolation_fn: &I,
    luma_fn: &LumaFn,
    kernel: &[[i32; 3 * 3]],
    fill: &T,
) where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
    LumaFn: Fn(&T) -> u32,
{
    let (width, height) = image.dimensions();

    if new_width > width {
        maximize_seam(image, new_width, interpolation_fn, luma_fn, kernel, fill);
    } else if new_width < width {
        minimize_seam(image, new_width, luma_fn, kernel);
    }

    image.rotate90_mut();

    if new_height > height {
        maximize_seam(image, new_height, interpolation_fn, luma_fn, kernel, fill);
    } else if new_height < height {
        minimize_seam(image, new_height, luma_fn, kernel);
    }

    image.rotate270_mut();
}

pub fn maximize_seam<T, I, LumaFn>(
    image: &mut FastImage<T>,
    new_width: usize,
    interpolation_fn: &I,
    luma_fn: &LumaFn,
    kernels: &[[i32; 3 * 3]],
    fill: &T,
) where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
    LumaFn: Fn(&T) -> u32,
{
    let current_width = image.dimensions().0;

    image.maximize(new_width, image.height, fill);

    let mut energy_image = FastImage::new(new_width, image.height, 0);
    energy(image, &mut energy_image, luma_fn, kernels, None);

    let mut waterfall_added = energy_image.clone();
    let mut last_seam_path = vec![0; image.height];

    for i in current_width..new_width {
        image.width = i - 1;
        energy_image.width = i - 1;
        waterfall_added.copy_from_image(&energy_image);

        add_waterfall(&mut waterfall_added);
        seam_path(&waterfall_added, &mut last_seam_path);
        shift_maximize(image, &last_seam_path, interpolation_fn);
        shift_maximize(&mut energy_image, &last_seam_path, &|_, _| 0);
        energy(
            image,
            &mut energy_image,
            luma_fn,
            kernels,
            Some((&last_seam_path, true)),
        );
    }
}

/// Splits the image top (0) to bottom (seam_path.len() - 1) along the seam and shifts the right part one pixel to the right.
///
/// interpolation_fn is a function which should take the left and right pixel which have been split, and return a color for the seam.
pub fn shift_maximize<I, T>(image: &mut FastImage<T>, seam_path: &[usize], interpolation_fn: &I)
where
    I: Fn(&T, &T) -> T,
    T: FastImagePixel,
{
    let (width, height) = image.dimensions();

    assert_eq!(seam_path.len(), height, "invalid seam path length");
    assert!(height > 1, "invalid image dimensions");
    assert!(
        image.width_stride > width,
        "the image must have an internal stride which is larger than its width"
    );

    let mut out_row_buf = vec![*unsafe { image.get_pixel_unchecked(0, 0) }; width];

    image
        .data
        .chunks_mut(image.width_stride)
        .zip(seam_path.into_iter().cloned())
        .for_each(|(row, seam_pos)| {
            unsafe {
                out_row_buf
                    .get_unchecked_mut(..)
                    .copy_from_slice(&row[..width]);

                out_row_buf.insert(
                    seam_pos,
                    if seam_pos != 0 {
                        let (left, right) = out_row_buf
                            .get_unchecked((seam_pos - 1)..(seam_pos + 1).min(width))
                            .split_at(1);

                        interpolation_fn(&left.get_unchecked(0), &right.get_unchecked(0))
                    } else {
                        *out_row_buf.get_unchecked(0)
                    },
                );

                row.get_unchecked_mut(..width + 1)
                    .copy_from_slice(&out_row_buf);
            }

            out_row_buf.pop();
        });

    image.width += 1;
}

pub fn minimize_seam<T, LumaFn>(
    image: &mut FastImage<T>,
    new_width: usize,
    luma_fn: &LumaFn,
    kernels: &[[i32; 3 * 3]],
) where
    LumaFn: Fn(&T) -> u32,
    T: FastImagePixel,
{
    let mut energy_image = FastImage::new(image.width, image.height, 0);
    energy(image, &mut energy_image, luma_fn, kernels, None);

    let mut waterfall_added = energy_image.clone();
    let mut last_seam_path = vec![0; image.height];

    for _ in (new_width..image.width).rev() {
        waterfall_added.copy_from_image(&energy_image);
        add_waterfall(&mut waterfall_added);
        seam_path(&waterfall_added, &mut last_seam_path);
        shift_minimize(image, &last_seam_path);
        shift_minimize(&mut energy_image, &last_seam_path);
        energy(
            image,
            &mut energy_image,
            luma_fn,
            kernels,
            Some((&last_seam_path, false)),
        );
    }

    /* *image = FastImage::from_data(
        energy_image.width,
        energy_image.height,
        energy_image
            .data
            .into_iter()
            .map(|p| unsafe {
                *(&Rgba([(p / 10000) as u8, (p / 10000) as u8, (p / 10000) as u8, 255]) as *const _
                    as *const _)
            })
            .collect(),
    )
    .unwrap();

    image.width_stride = energy_image.width_stride;*/
}

pub fn shift_minimize<T>(image: &mut FastImage<T>, seam_path: &[usize])
where
    T: FastImagePixel,
{
    assert_eq!(seam_path.len(), image.height, "invalid seam path length");
    let (width, height) = image.dimensions();
    assert!(image.width >= 1, "image dimensions too small");

    for (row_idx, &seam_pos) in (0..height).zip(seam_path.into_iter()) {
        if seam_pos < width {
            unsafe {
                image
                    .get_row_unchecked_mut(seam_pos..width, row_idx)
                    .rotate_left(1);
            }
        }
    }

    image.minimize(width - 1, height);
}

/// Traverses the image's lines from bottom to top to find the most energetic seam.
pub fn seam_path(energy_image: &FastImage<u32>, seam: &mut [usize]) {
    let (width, height) = energy_image.dimensions();
    assert!(
        energy_image.height >= 2 && energy_image.width >= 3,
        "image dimensions too small"
    );
    assert_eq!(seam.len(), energy_image.height, "invalid iamge dimensions");

    // find the least energetic pixel in the bottom row of pixels
    let mut least_energetic_idx = unsafe { energy_image.get_row_unchecked(0..width, height - 1) }
        .iter()
        .enumerate()
        .min_by_key(|(_, p)| **p)
        .unwrap()
        .0;

    seam[height as usize - 1] = least_energetic_idx;

    // find a seam from bottom to top
    for y in (0..height - 1).rev() {
        unsafe {
            let range = least_energetic_idx.saturating_sub(1)..if least_energetic_idx < width - 1 {
                least_energetic_idx + 2
            } else {
                width
            };

            least_energetic_idx = range.start
                + energy_image
                    .get_row_unchecked(range, y + 1)
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| **p)
                    .unwrap()
                    .0;

            seam[y as usize] = least_energetic_idx;
        }
    }
}

/// Iterates over every line from top to bottom. Each pixel in a line adds the value of the lowest of the three pixels above to itself.
///
/// ## Example
/// ```rust
/// //  1  2  3
/// // 10 10 10
/// let mut image = FastImage::from_data(3, 2, vec![1u16, 2, 3, 10, 10, 10]);
/// seam_carving::add_waterfall(&mut image);
/// // we check for the pixel in the center of the second line
/// assert_eq!(image.get_pixel(1, 1), 10 + 1);
/// ```
pub fn add_waterfall(image: &mut FastImage<u32>) {
    assert!(
        image.width >= 3 && image.height >= 1,
        "image dimensions too small"
    );

    let (width, height) = image.dimensions();

    for y in 1..height {
        // P1P2P3
        //   CP

        // left border
        unsafe {
            let two_above = image.get_row_unchecked(0..2, y - 1);
            let new_value = image
                .get_pixel_unchecked(0, y)
                .saturating_add(*two_above.get_unchecked(0).min(two_above.get_unchecked(1)));

            image.set_pixel_unchecked(0, y, new_value);
        }

        for x in 1..width - 1 {
            unsafe {
                let three_above = image.get_row_unchecked(x - 1..x + 2, y - 1);

                let new_value = image.get_pixel_unchecked(x, y).saturating_add(
                    *three_above.get_unchecked(0).min(
                        three_above
                            .get_unchecked(1)
                            .min(three_above.get_unchecked(2)),
                    ),
                );

                // find the pixel with the lowest value of the three pixels above
                image.set_pixel_unchecked(x, y, new_value);
            }
        }

        // right border
        unsafe {
            let two_above = image.get_row_unchecked(width - 2..width, y - 1);

            let new_value = image
                .get_pixel_unchecked(width - 1, y)
                .saturating_add(*two_above.get_unchecked(0).min(two_above.get_unchecked(1)));

            image.set_pixel_unchecked(width - 1, y, new_value);
        }
    }
}

/// Applies the given function `luma_fn` to convert each pixel into its luma representation and then applies the given 3x3 filtering kernel `kernel` to convert the luma representation to an energy image.
pub fn energy<T, LumaFn>(
    image: &FastImage<T>,
    output: &mut FastImage<u32>,
    luma_fn: &LumaFn,
    kernels: &[[i32; 3 * 3]],
    seam: Option<(&[usize], bool)>,
) where
    LumaFn: Fn(&T) -> u32,
    T: FastImagePixel,
{
    let (width, height) = image.dimensions();
    let width_stride = image.width_stride;

    let mut image_luma: FastImage<u32> = FastImage::new(width, height, 0);

    let mut temp_buf = vec![0; width];
    for row_idx in 0..image_luma.height {
        unsafe {
            temp_buf
                .iter_mut()
                .zip(image.get_row_unchecked(0..width, row_idx).iter())
                .for_each(|(luma, rgb)| *luma = luma_fn(rgb));

            image_luma.set_row_unchecked(0..width, row_idx, &temp_buf);
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

    if let Some((seam, wide)) = seam {
        // operate in seam mode: only calculate the pixels lying to the side of, and on, the seam

        for kernel in kernels {
            for ((kernel_offset, kx, ky), kernel) in
                kernel_offsets.iter().cloned().zip(kernel.iter().cloned())
            {
                for (row_idx, &seam_idx) in
                    (ky.max(0)..(height as i32 + ky.min(0))).zip(seam.into_iter())
                {
                    for x in kx.max(seam_idx as i32 - 1).max(0) as usize
                        ..(seam_idx as i32 + if wide { 3 } else { 2 }).min(width as i32 + kx.min(0))
                            as usize
                    {
                        unsafe {
                            *(output.data.get_unchecked_mut(
                                (row_idx as usize * output.width_stride + x) as usize,
                            ) as *mut u32 as *mut i32) += *image_luma.data.get_unchecked(
                                ((row_idx as usize * image_luma.width_stride + x) as i32
                                    + kernel_offset) as usize,
                            ) as i32
                                * kernel;
                        }
                    }
                }
            }

            unsafe {
                output.data.iter_mut().for_each(|x| {
                    *(x as *mut u32 as *mut i32) = (*(x as *mut u32 as *mut i32)).abs()
                });
            }
        }
    } else {
        for kernel in kernels {
            for ((kernel_offset, kx, ky), kernel) in
                kernel_offsets.iter().cloned().zip(kernel.iter().cloned())
            {
                for row_idx in ky.max(0)..(height as i32 + ky.min(0)) {
                    for x in kx.max(0) as usize..(width as i32 + kx.min(0)) as usize {
                        unsafe {
                            *(output.data.get_unchecked_mut(
                                (row_idx as usize * output.width_stride + x) as usize,
                            ) as *mut u32 as *mut i32) += *image_luma.data.get_unchecked(
                                ((row_idx as usize * image_luma.width_stride + x) as i32
                                    + kernel_offset) as usize,
                            ) as i32
                                * kernel;
                        }
                    }
                }
            }

            unsafe {
                output
                    .data
                    .iter_mut()
                    .for_each(|x| *x = (*(&*x as *const u32 as *const i32)).abs() as u32);
            }
        }
    }

    /*
    // squared error algorithm
    let kernel_offsets = [
        (width as i32, 0, -1),
        (1, -1, 0),
        (-1, 1, 0),
        (-(width as i32), 0, 1),
    ];

    for (kernel_offset, kx, ky) in kernel_offsets.iter().cloned() {
        for row_idx in (ky.max(0)..(height as i32 + ky.min(0))).map(|row| row as usize * width) {
            for x in kx.max(0) as usize..(width as i32 + kx.min(0)) as usize {
                unsafe {
                    *output.data.get_unchecked_mut((row_idx + x) as usize) +=
                        (*image_luma.data.get_unchecked((row_idx + x) as usize) as i32
                            - *image_luma
                                .data
                                .get_unchecked(((row_idx + x) as i32 + kernel_offset) as usize)
                                as i32)
                            .pow(2);
                }
            }
        }
    }*/
}
