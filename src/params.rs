use crate::custom_rand::{choice, rndtri, rndtri_f};

pub struct Params {
    body_curve_type: i32,
    body_curve_amount: f64,
    body_length: i32,
    body_height: i32,
    scale_type: i32,
    scale_scale: f64,
    pattern_type: i32,
    pattern_scale: f64,
    dorsal_texture_type: i32,
    dorsal_type: i32,
    dorsal_length: i32,
    dorsal_start: i32,
    dorsal_end: i32,
    wing_texture_type: i32,
    wing_type: i32,
    wing_start: i32,
    wing_end: i32,
    wing_y: f64,
    wing_length: i32,
    wing_width: i32,
    pelvic_start: i32,
    pelvic_end: i32,
    pelvic_length: i32,
    pelvic_type: i32,
    pelvic_texture_type: i32,
    anal_start: i32,
    anal_end: i32,
    anal_length: i32,
    anal_type: i32,
    anal_texture_type: i32,
    tail_type: i32,
    tail_length: i32,
    finlet_type: i32,
    neck_type: i32,
    nose_height: i32,
    mouth_size: i32,
    head_length: i32,
    head_texture_amount: i32,
    has_moustache: i32,
    moustache_length: i32,
    has_beard: i32,
    has_teeth: i32,
    teeth_length: i32,
    teeth_space: f64,
    beard_length: i32,
    eye_type: i32,
    eye_size: i32,
    jaw_size: f64,
    jaw_open: i32,
}
impl Params {
    pub fn default() -> Self {
        Params {
            body_curve_type: 0,
            body_curve_amount: 0.85,
            body_length: 350,
            body_height: 90,
            scale_type: 1,
            scale_scale: 1.,
            pattern_type: 3,
            pattern_scale: 1.,
            dorsal_texture_type: 1,
            dorsal_type: 0,
            dorsal_length: 100,
            dorsal_start: 8,
            dorsal_end: 27,
            wing_texture_type: 0,
            wing_type: 0,
            wing_start: 6,
            wing_end: 6,
            wing_y: 0.7,
            wing_length: 130,
            wing_width: 10,
            pelvic_start: 9,
            pelvic_end: 14,
            pelvic_length: 85,
            pelvic_type: 0,
            pelvic_texture_type: 0,
            anal_start: 19,
            anal_end: 29,
            anal_length: 50,
            anal_type: 0,
            anal_texture_type: 0,
            tail_type: 0,
            tail_length: 75,
            finlet_type: 0,
            neck_type: 0,
            nose_height: 0,
            mouth_size: 8,
            head_length: 30,
            head_texture_amount: 60,
            has_moustache: 1,
            moustache_length: 10,
            has_beard: 0,
            has_teeth: 1,
            teeth_length: 8,
            teeth_space: 3.5,
            beard_length: 30,
            eye_type: 1,
            eye_size: 10,
            jaw_size: 1.,
            jaw_open: 1,
        }
    }
}

pub fn generate_params() -> Params {
    let mut arg = Params::default();
    arg.body_curve_type = *choice(&[0, 1], None);
    arg.body_curve_amount = rndtri_f(0.5, 0.85, 0.98);
    arg.body_length = rndtri(200, 350, 420);
    arg.body_height = rndtri(45, 90, 150);
    arg.scale_type = *choice(&[0, 1, 2, 3], None);
    arg.scale_scale = rndtri_f(0.8, 1., 1.5);
    arg.pattern_type = *choice(&[0, 1, 2, 3, 4], None);
    arg.pattern_scale = rndtri_f(0.5, 1., 2.);
    arg.dorsal_texture_type = *choice(&[0, 1], None);
    arg.dorsal_type = *choice(&[0, 1], None);
    arg.dorsal_length = rndtri(30, 90, 180);
    if (arg.dorsal_type == 0) {
        arg.dorsal_start = !!rndtri(7, 8, 15);
        arg.dorsal_end = !!rndtri(20, 27, 28);
    } else {
        arg.dorsal_start = !!rndtri(11, 12, 16);
        arg.dorsal_end = !!rndtri(19, 21, 24);
    }
    arg.wing_texture_type = *choice(&[0, 1], None);
    arg.wing_type = *choice(&[0, 1], None);
    if (arg.wing_type == 0) {
        arg.wing_length = rndtri(40, 130, 200);
    } else {
        arg.wing_length = rndtri(40, 150, 350);
    }
    if (arg.wing_texture_type == 0) {
        arg.wing_width = rndtri(7, 10, 20);
        arg.wing_y = rndtri_f(0.45, 0.7, 0.85);
    } else {
        arg.wing_width = rndtri(20, 30, 50);
        arg.wing_y = rndtri_f(0.45, 0.65, 0.75);
    }

    arg.wing_start = !!rndtri(5, 6, 8);
    arg.wing_end = !!rndtri(5, 6, 8);

    arg.pelvic_texture_type = if arg.dorsal_texture_type != 0 {
        *choice(&[0, 1], None)
    } else {
        0
    };
    arg.pelvic_type = *choice(&[0, 1], None);
    arg.pelvic_length = rndtri(30, 85, 140);
    if (arg.pelvic_type == 0) {
        arg.pelvic_start = !!rndtri(7, 9, 11);
        arg.pelvic_end = !!rndtri(13, 14, 15);
    } else {
        arg.pelvic_start = !!rndtri(7, 9, 12);
        arg.pelvic_end = arg.pelvic_start + 2;
    }

    arg.anal_texture_type = if arg.dorsal_texture_type != 0 {
        *choice(&[0, 1], None)
    } else {
        0
    };
    arg.anal_type = *choice(&[0, 1], None);
    arg.anal_length = rndtri(20, 50, 80);
    arg.anal_start = !!rndtri(16, 19, 23);
    arg.anal_end = !!rndtri(25, 29, 31);

    arg.tail_type = *choice(&[0, 1, 2, 3, 4, 5], None);
    arg.tail_length = rndtri(50, 75, 180);

    arg.finlet_type = *choice(&[0, 1, 2, 3], None);

    arg.neck_type = *choice(&[0, 1], None);
    arg.nose_height = rndtri(-50, 0, 35);
    arg.head_length = rndtri(20, 30, 35);
    arg.mouth_size = !!rndtri(6, 8, 11);

    arg.head_texture_amount = !!rndtri(30, 60, 160);
    arg.has_moustache = *choice(&[0, 0, 0, 1], None);
    arg.has_beard = *choice(&[0, 0, 0, 0, 0, 1], None);
    arg.moustache_length = !!rndtri(10, 20, 40);
    arg.beard_length = !!rndtri(20, 30, 50);

    arg.eye_type = *choice(&[0, 1], None);
    arg.eye_size = rndtri(8, 10, 28); //arg.body_height/6//Math.min(arg.body_height/6,rndtri(8,10,30));

    arg.jaw_size = rndtri_f(0.7, 1., 1.4);

    arg.has_teeth = *choice(&[0, 1, 1], None);
    arg.teeth_length = rndtri(5, 8, 15);
    arg.teeth_space = rndtri_f(3., 3.5, 6.);

    return arg;
}
