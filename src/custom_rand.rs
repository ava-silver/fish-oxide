use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{
    f64::consts::PI,
    sync::{
        atomic::{AtomicU64, Ordering},
        LazyLock,
    },
};

static JSR: AtomicU64 = AtomicU64::new(0x5EED);

pub fn str_to_seed(str: String) -> u32 {
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

pub fn rand() -> f64 {
    return thread_rng().gen();
    // let mut reg = JSR.load(Ordering::Acquire);
    // reg ^= reg << 17;
    // reg ^= reg >> 13;
    // reg ^= reg << 5;
    // JSR.store(reg, Ordering::Release);
    // return (reg >> 0) as f64 / 4294967295.;
}

pub fn seed_rand(seed: u64) {
    JSR.store(seed, Ordering::SeqCst);
}

const PERLIN_YWRAPB: u64 = 4;
const PERLIN_YWRAP: u64 = 1 << PERLIN_YWRAPB;
const PERLIN_ZWRAPB: u64 = 8;
const PERLIN_ZWRAP: u64 = 1 << PERLIN_ZWRAPB;
const PERLIN_SIZE: u64 = 4095;
const PERLIN_OCTAVES: u64 = 4;
const PERLIN_AMP_FALLOFF: f64 = 0.5;
pub fn scaled_cosine(i: f64) -> f64 {
    return 0.5 * (1.0 - f64::cos(i * PI));
}

static PERLIN: LazyLock<Vec<f64>> = LazyLock::new(|| (0..=PERLIN_SIZE).map(|_| rand()).collect());

pub fn noise(mut x: f64, y_opt: Option<f64>, z_opt: Option<f64>) -> f64 {
    x = x.abs();
    let y = y_opt.unwrap_or(0.).abs();
    let z = z_opt.unwrap_or(0.).abs();

    let mut xi = x.floor() as u64;
    let mut yi = y.floor() as u64;
    let mut zi = z.floor() as u64;

    let mut xf = x - (xi as f64);
    let mut yf = y - (yi as f64);
    let mut zf = z - (zi as f64);

    let mut rxf;
    let mut ryf;

    let mut r = 0.;
    let mut ampl = 0.5;

    let mut n1;
    let mut n2;
    let mut n3;
    for _ in 0..PERLIN_OCTAVES {
        let mut of = xi + ((yi) << PERLIN_YWRAPB) + ((zi) << PERLIN_ZWRAPB);
        rxf = scaled_cosine(xf);
        ryf = scaled_cosine(yf);
        n1 = PERLIN[(of & PERLIN_SIZE) as usize];
        n1 += rxf * (PERLIN[((of + 1) & PERLIN_SIZE) as usize] - n1);
        n2 = PERLIN[((of + PERLIN_YWRAP) & PERLIN_SIZE) as usize];
        n2 += rxf * (PERLIN[((of + PERLIN_YWRAP + 1) & PERLIN_SIZE) as usize] - n2);
        n1 += ryf * (n2 - n1);
        of += PERLIN_ZWRAP;
        n2 = PERLIN[(of & PERLIN_SIZE) as usize];
        n2 += rxf * (PERLIN[((of + 1) & PERLIN_SIZE) as usize] - n2);
        n3 = PERLIN[((of + PERLIN_YWRAP) & PERLIN_SIZE) as usize];
        n3 += rxf * (PERLIN[((of + PERLIN_YWRAP + 1) & PERLIN_SIZE) as usize] - n3);
        n2 += ryf * (n3 - n2);
        n1 += scaled_cosine(zf) * (n2 - n1);
        r += n1 * ampl;
        ampl *= PERLIN_AMP_FALLOFF;
        xi <<= 1;
        xf *= 2.;
        yi <<= 1;
        yf *= 2.;
        zi <<= 1;
        zf *= 2.;

        if xf >= 1.0 {
            xi += 1;
            xf -= 1.;
        }
        if yf >= 1.0 {
            yi += 1;
            yf -= 1.;
        }
        if zf >= 1.0 {
            zi += 1;
            zf -= 1.;
        }
    }
    return r;
}

pub fn choice<'a, T>(opts: &'a [T], percs_opt: Option<&[f64]>) -> &'a T {
    return opts.choose(&mut thread_rng()).unwrap();
    let default_percs = opts.iter().map(|_| 1.).collect::<Vec<_>>();
    let percs = percs_opt.unwrap_or(&default_percs);
    let mut s = percs.iter().sum();
    let mut r = rand() * s;
    s = 0.;
    for i in 0..percs.len() {
        s += percs[i];
        if r <= s {
            return &opts[i];
        }
    }
    unreachable!();
}

pub fn rndtri(a: i64, b: i64, c: i64) -> i64 {
    let s0 = (b - a) / 2;
    let s1 = (c - b) / 2;
    let s = s0 + s1;
    let r = rand() as i64 * s;
    if r < s0 {
        //d * d/(b-a) / 2 = r;
        let d = ((2 * r * (b - a)) as f32).sqrt();
        return a + d as i64;
    }
    //d * d/(c-b) / 2 = s-r;
    let d = ((2 * (s - r) * (c - b)) as f32).sqrt();
    return c - d as i64;
}
pub fn rndtri_f(a: f64, b: f64, c: f64) -> f64 {
    let s0 = (b - a) / 2.;
    let s1 = (c - b) / 2.;
    let s = s0 + s1;
    let r = rand() as f64 * s;
    if r < s0 {
        //d * d/(b-a) / 2 = r;
        let d = (2. * r * (b - a)).sqrt();
        return a + d;
    }
    //d * d/(c-b) / 2 = s-r;
    let d = (2. * (s - r) * (c - b)).sqrt();
    return c - d;
}

pub fn deviate(n: f64) -> f64 {
    return (rand() as f64 * 2. * n) - n;
}
