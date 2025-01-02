use clap::ValueEnum;
use custom_rand::{seed_rand, str_to_seed};
use draw::{cleanup, draw_svg, fish, reframe};

pub use crate::params::generate_params;
pub use crate::params::Params;

pub mod custom_rand;
pub mod draw;
pub mod geometry;
pub mod hershey;
pub mod params;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Format {
    Svg,
    Json,
    Smil,
    Csv,
    Ps,
}

const DEFAULT_SEED: &str = "x11:11 make a fish";

pub fn generate(
    format: Format,
    seed: String,
    caption: Option<String>,
    _speed_opt: Option<f64>,
    pad_opt: Option<f64>,
    fish_opts: Option<params::Params>,
) -> String {
    seed_rand(str_to_seed(seed.to_owned()).into());

    // let mut speed = 0.005 / speed_opt.unwrap_or(1.);
    let drawing = fish(fish_opts.unwrap_or_else(generate_params));
    let polylines = cleanup(reframe(drawing, pad_opt, Some(caption.unwrap_or(seed))));

    match format {
        Format::Svg => draw_svg(polylines),
        Format::Json => serde_json::to_string(&polylines).unwrap(),
        Format::Smil => {
            todo!("Smil format is not yet supported")

            // draw_svg_anim(polylines, speed)
        }
        Format::Csv => polylines
            .iter()
            .map(|x| {
                x.iter()
                    .map(|z| format!("{:?}", *z))
                    .collect::<Vec<String>>()
                    .join(",")
            })
            .collect::<Vec<String>>()
            .join("\n"),
        Format::Ps => {
            todo!("Ps format is not yet supported")
            // draw_ps(polylines)
        }
    }
}

pub fn generate_svg() -> String {
    generate(
        Format::Svg,
        DEFAULT_SEED.to_string(),
        Some(DEFAULT_SEED.to_string()),
        None,
        Some(20.),
        None,
    )
}

pub fn generate_json() -> String {
    generate(
        Format::Json,
        DEFAULT_SEED.to_string(),
        Some(DEFAULT_SEED.to_string()),
        None,
        Some(20.),
        None,
    )
}

pub fn generate_csv() -> String {
    generate(
        Format::Csv,
        DEFAULT_SEED.to_string(),
        Some(DEFAULT_SEED.to_string()),
        None,
        Some(20.),
        None,
    )
}
