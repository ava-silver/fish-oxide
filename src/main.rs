#![allow(dead_code, unused)]
use clap::{command, Parser, ValueEnum};
use core::f64;
use core::f64::consts::{E, PI};
use draw::{cleanup, reframe};
use geometry::{dist, get_boundingbox, trsl_poly, Polyline};
use hershey::compile_hershey;
use params::generate_params;
use rand::{thread_rng, Rng};
use regex::Regex;
use std::cmp::Ordering;
use std::iter::successors;
use std::sync::LazyLock;
use std::{collections::HashMap, default};

pub mod custom_rand;
pub mod draw;
pub mod geometry;
pub mod hershey;
pub mod params;

use custom_rand::{choice, rand, rndtri, rndtri_f, seed_rand};

fn str_to_seed(str: String) -> u32 {
    let mut n = 1;
    for (i, c) in str.chars().enumerate() {
        let x = (c as u32) + 1;
        n ^= x << (7 + (i % 5));
        // if (i % 2){
        n ^= n << 17;
        n ^= n >> 13;
        n ^= n << 5;
        // }
        n = (n >> 0) % 4294967295;
    }
    return n;
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Format {
    Svg,
    Json,
    Smil,
    Csv,
    Ps,
}

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
    let mut seed = args.seed.unwrap_or_else(|| {
        seed_rand(!!(thread_rng().gen::<u64>()));
        "binomen()".to_string()
    });
    seed_rand(str_to_seed(seed).into());

    let format = args.format.unwrap_or(Format::Svg);
    let mut speed = 0.005;
    speed = speed / args.speed.unwrap_or(1.);
    let drawing = todo!("fish(generate_params())");
    seed += ".";
    let polylines = cleanup(reframe(drawing, Some(20.), Some((seed))));

    println!(
        "{}",
        match format {
            Format::Svg => {
                todo!()
                // draw_svg(polylines)
            }
            Format::Json => {
                serde_json::to_string(&polylines).unwrap()
            }
            Format::Smil => {
                todo!()

                // draw_svg_anim(polylines, speed)
            }
            Format::Csv => {
                polylines
                    .iter()
                    .map(|x| {
                        x.iter()
                            .map(|z| format!("{:?}", *z))
                            .collect::<Vec<String>>()
                            .join(",")
                    })
                    .collect::<Vec<String>>()
                    .join("\n")
            }
            Format::Ps => {
                todo!()
                // draw_ps(polylines)
            }
        }
    );
}
