use clap::{App, Arg};
use gifski::{progress::ProgressBar, Settings};
use image::{DynamicImage, FilterType, GenericImageView};
use imgref::Img;
use parking_lot::RwLock;
use rayon::ThreadPoolBuilder;
use rgb::RGBA8;
use std::{fs::OpenOptions, sync::Arc, thread, time::Instant};

pub mod seam_carving;

fn main() {
    let clap = App::new("Animated seam carving")
        .author("Authors: DasEtwas, Friz64")
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .value_name("FILE")
                .help("Input file path")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .value_name("FILE")
                .default_value("output.gif")
                .help("Output file")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("length")
                .short("l")
                .long("length")
                .value_name("SECONDS")
                .default_value("1.0")
                .help("GIF length in seconds (float)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("scale")
                .short("s")
                .long("scale")
                .value_name("PERCENT")
                .default_value("10")
                .help("Final scale of the last frame in percent (float)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("fps")
                .short("f")
                .long("fps")
                .value_name("FPS")
                .default_value("20")
                .help("GIF FPS, is converted into GIF units (10ms) (float)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("quality")
                .short("q")
                .long("quality")
                .value_name("QUALITY")
                .default_value("30")
                .help("GIF Quality [1 - 100]")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("single_threaded")
                .short("t")
                .long("single")
                .help("If only one thread should be used."),
        )
        .get_matches();

    let input = clap.value_of("input").unwrap();
    let output = clap.value_of("output").unwrap();
    let single = clap.occurrences_of("single_threaded") != 0;
    let length: f32 = match clap.value_of("length").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing fps: {}", err);
            return;
        }
    };
    let scale: f32 = match clap.value_of("scale").unwrap().parse::<f32>() {
        Ok(val) => val / 100.0,
        Err(err) => {
            println!("Error while parsing scale: {}", err);
            return;
        }
    };
    let fps: f32 = match clap.value_of("fps").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing fps: {}", err);
            return;
        }
    };
    let quality: u8 = match clap.value_of("quality").unwrap().parse() {
        Ok(val) => val,
        Err(err) => {
            println!("Error while parsing quality: {}", err);
            return;
        }
    };

    let delay = (((1.0 / fps) * 100.0).floor() as u16).max(1);
    let fps = 1.0 / (delay as f32 / 100.0);
    let frames = fps * length;

    let image = match image::open(input) {
        Ok(val) => val,
        Err(err) => {
            println!("Failed to open input: {}", err);
            return;
        }
    };

    let dimensions = image.dimensions();
    let width = dimensions.0 as f32;
    let height = dimensions.1 as f32;

    if width * scale < 3.0 {
        println!("target width is smaller than 3! will clamp to 3 (try adjusting scale)")
    }
    if height * scale < 3.0 {
        println!("target height is smaller than 3! will clamp to 3 (try adjusting scale)")
    }

    let (mut collector, writer) = gifski::new(Settings {
        width: Some(dimensions.0),
        height: Some(dimensions.1),
        quality,
        once: false,
        fast: false,
    })
    .expect("Failed to create encoder");

    println!("Carving seams..");

    let start = Instant::now();

    let upscaling = scale > 1.0;

    if single {
        thread::spawn(move || {
            let mut last_frame = image;

            (0..frames as usize).for_each(|i| {
                let new_width = (lerp(
                    width,
                    width * scale,
                    i as f32 / (frames as usize - 1).max(1) as f32,
                ) as u32)
                    .max(3);
                let new_height = (lerp(
                    height,
                    height * scale,
                    i as f32 / (frames as usize - 1).max(1) as f32,
                ) as u32)
                    .max(3);

                let frame_image =
                    seam_carving::easy_resize(&last_frame, new_width as usize, new_height as usize);

                last_frame = frame_image.clone();

                let frame = if upscaling {
                    image_to_frame(&frame_image.resize_exact(
                        (width * scale) as u32,
                        (height * scale) as u32,
                        FilterType::Nearest,
                    ))
                } else {
                    image_to_frame(&frame_image.resize_exact(
                        width as u32,
                        height as u32,
                        FilterType::Nearest,
                    ))
                };

                collector
                    .add_frame_rgba(i, frame, delay)
                    .expect("Failed to add frame");
            });
        });
    } else {
        let last_image = Arc::new(RwLock::new(image));

        let (gif_renderer_sender, gif_renderer_receiver) =
            crossbeam_channel::bounded((frames as usize).min(300));

        thread::spawn({
            let last_image = last_image.clone();

            move || {
                let pool = ThreadPoolBuilder::new()
                    .num_threads(num_cpus::get())
                    .build()
                    .unwrap();

                (0..frames as usize).for_each(|i| {
                    pool.spawn_fifo({
                        let last_image = last_image.clone();
                        let gif_renderer_sender = gif_renderer_sender.clone();

                        move || {
                            let new_width = (lerp(
                                width,
                                width * scale,
                                i as f32 / (frames as usize - 1).max(1) as f32,
                            ) as u32)
                                .max(3);

                            let new_height = (lerp(
                                height,
                                height * scale,
                                i as f32 / (frames as usize - 1).max(1) as f32,
                            ) as u32)
                                .max(3);

                            let current_image = last_image.read().clone();

                            let frame_image = seam_carving::easy_resize(
                                &current_image,
                                new_width as usize,
                                new_height as usize,
                            );

                            {
                                let mut last_image = last_image.write();
                                if upscaling {
                                    if frame_image.width() > last_image.width() {
                                        *last_image = frame_image.clone();
                                    }
                                } else {
                                    if frame_image.width() < last_image.width() {
                                        *last_image = frame_image.clone();
                                    }
                                }
                            }

                            let frame = if upscaling {
                                image_to_frame(&frame_image.resize_exact(
                                    (width * scale) as u32,
                                    (height * scale) as u32,
                                    FilterType::Nearest,
                                ))
                            } else {
                                image_to_frame(&frame_image.resize_exact(
                                    width as u32,
                                    height as u32,
                                    FilterType::Nearest,
                                ))
                            };

                            gif_renderer_sender
                                .send((i, frame))
                                .expect("channel disconnected");
                        }
                    })
                });
            }
        });

        thread::spawn(move || {
            while let Ok((i, frame)) = gif_renderer_receiver.recv() {
                collector
                    .add_frame_rgba(i, frame, delay)
                    .expect("Failed to add frame");
            }
        });
    }

    let output_result = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output);

    let output = match output_result {
        Ok(val) => val,
        Err(err) => {
            println!("Failed to open output: {}", err);
            return;
        }
    };

    let mut progress_bar = ProgressBar::new(frames.round() as u64);

    writer
        .write(output, &mut progress_bar)
        .expect("Failed to write output");

    println!("Done in {:?}!", start.elapsed());
}

fn lerp(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

fn image_to_frame(image: &DynamicImage) -> Img<Vec<RGBA8>> {
    let (width, height) = image.dimensions();

    let container: Vec<RGBA8> = image
        .to_rgba()
        .into_raw()
        .chunks(4)
        .map(|chunk| RGBA8::new(chunk[0], chunk[1], chunk[2], chunk[3]))
        .collect();

    Img::new(container, width as usize, height as usize)
}
