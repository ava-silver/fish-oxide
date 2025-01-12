#![allow(dead_code, unused)]
use clap::{command, Parser};
use core::f64;
use fish_oxide::{generate, Format};
use hershey::binomen;

pub mod custom_rand;
pub mod draw;
pub mod geometry;
pub mod hershey;
pub mod params;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Output format, defaults to svg
    #[arg(short, long)]
    format: Option<Format>,
    /// Random seed
    #[arg(long)]
    seed: Option<String>,
    /// Animation speed
    #[arg(long)]
    speed: Option<f64>,
}

fn main() {
    let args = Cli::parse();
    let seed = args.seed.unwrap_or(binomen());
    let format = args.format.unwrap_or(Format::Svg);

    println!(
        "{}",
        generate(
            format,
            seed.clone(),
            Some(seed),
            args.speed,
            Some(20.),
            None
        )
    );
}
