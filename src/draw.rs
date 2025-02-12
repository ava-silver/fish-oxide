use std::{
    f64::consts::{E, PI},
    iter::successors,
    rc::Rc,
};

use crate::{
    custom_rand::{deviate, noise, rand},
    geometry::{
        binclip, binclip_multi, clip, clip_multi, dist, fill_shape, flat, get_boundingbox, lerp,
        lerp2d, pattern_dot, patternshade_shape, poly_union, pow, pt_seg_dist, resample, scl_poly,
        shade_shape, shr_poly, smalldot_shape, trsl_poly, vein_shape, Point, Polyline, PolylineOps,
    },
    hershey::compile_hershey,
    params::Params,
};

const SVG_PREAMBLE: &str = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"520\" height=\"320\">
<rect x=\"0\" y=\"0\" width=\"520\" height=\"320\" fill=\"floralwhite\"/>
<rect x=\"10\" y=\"10\" width=\"500\" height=\"300\" stroke=\"black\" stroke-width=\"1\" fill=\"none\"/>
<path stroke=\"black\" stroke-width=\"1\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\" d=\"";

pub fn draw_svg(polylines: Vec<Polyline>) -> String {
    SVG_PREAMBLE.to_string()
        + &polylines
            .iter()
            .map(|line| {
                "\nM ".to_string()
                    + &line
                        .iter()
                        .map(|(x, y)| {
                            format!(
                                "{} {} ",
                                (((x + 10.) * 100.).trunc()) / 100.,
                                ((y + 10.) * 100.).trunc() / 100.
                            )
                        })
                        .collect::<String>()
            })
            .collect::<String>()
        + "\n\"/></svg>"
}
/*

pub pub fn draw_svg_anim(polylines,speed){
  let o = `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320">`;
  o += `<rect x="0" y="0" width="520" height="320" fill="floralwhite"/><rect x="10" y="10" width="500" height="300" stroke="black" stroke-width="1" fill="none"/>`
  let lengths = [];
  let acc_lengths = [];
  let total_l = 0;
  for (let i = 0; i < polylines.length; i++){
    let l = 0;
    for (let j = 1; j < polylines[i].length; j++){
      l += f64::hypot(
        polylines[i][j-1].0-polylines[i][j].0,
        polylines[i][j-1].1-polylines[i][j].1
      );
    }
    lengths.push(l);
    acc_lengths.push(total_l);
    total_l+=l;
  }
  for (let i = 0; i < polylines.length; i++){
    let l = lengths[i];
    o += `
    <path
      stroke="black"
      stroke-width="1"
      fill="none"
      stroke-dasharray="${l}"
      stroke-dashoffset="${l}"
      d="M`;
    for (let j = 0; j < polylines[i].length; j++){
      o += polylines[i][j] + ' ';
    }
    let t = speed*l;
    o += `">
    <animate id="a${i}"
      attributeName="stroke-dashoffset"
      fill="freeze"
      from="${l}" to="${0}" dur="${t}s"
      begin="${(acc_lengths[i])*speed}s;a${i}.end+${8+speed*total_l-t}s"/>
    />
    <animate id="b${i}"
      attributeName="stroke-dashoffset"
      fill="freeze"
      from="${0}" to="${l}" dur="${3}s"
      begin="${5+speed*total_l}s;b${i}.end+${5+speed*total_l}s"/>
    />
    </path>`;
  }
  o += `</svg>`;
  return o;
}

pub pub fn draw_ps(polylines){
  let o = `%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 520 320
1 setlinewidth
0.5 0.5 translate
/m /moveto load def
/l /lineto load def
/F /stroke load def
%%EndPageSetup
10 10 m
510 10 l
510 310 l
10 310 l
closepath
F
`;
  for (let i = 0; i < polylines.length; i++){
    for (let j = 0; j < polylines[i].length; j++){
      let [x,y] = polylines[i][j];
      o += `${(~~((x+10)*100)) /100} ${(~~((310-y)*100)) /100} `;
      if (j == 0) {
        o += `m\n`;
      } else {
        o += `l\n`;
      }
    }
    o += `F\n\n`;
  }
  return o;
}
*/
fn squama_mask(w: f64, h: f64) -> Polyline {
    let mut p = vec![];
    let n = 7;
    for i in 0..n {
        let t = i as f64 / n as f64;
        let a = t * PI * 2.;
        let x = -pow(a.cos(), 1.3) * w;
        let y = pow(a.sin(), 1.3) * h;
        p.push((x, y));
    }
    return p;
}

fn squama(w: f64, h: f64, m_opt: Option<usize>) -> Vec<Polyline> {
    let m = m_opt.unwrap_or(3);
    let mut p = vec![];
    let n = 8;
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let a = t * PI + PI / 2.;
        let cos = a.cos();
        let sin = a.sin();
        let x = cos.abs().powf(1.4).copysign(-cos) * w;
        let y = sin.abs().powf(1.4).copysign(sin) * h;
        p.push((x, y));
    }
    let mut q = vec![p];
    for i in 0..m {
        let t = i as f64 / (m - 1) as f64;
        q.push(vec![
            (
                -w * 0.3 + (rand() - 0.5),
                -h * 0.2 + t * h * 0.4 + (rand() - 0.5),
            ),
            (
                w * 0.5 + (rand() - 0.5),
                -h * 0.3 + t * h * 0.6 + (rand() - 0.5),
            ),
        ]);
    }
    return q;
}

pub fn squama_mesh(
    m: usize,
    n: usize,
    (uw, uh): Point,
    squama_func: Box<dyn Fn(Point, Point) -> Vec<Polyline>>,
    (noise_x, noise_y): Point,
    interclip_opt: Option<bool>,
) -> Vec<Polyline> {
    let interclip = interclip_opt.unwrap_or(true);
    let mut clipper: Option<Polyline> = None;

    let mut pts = vec![];
    for i in 0..n {
        for j in 0..m {
            let x = j as f64 * uw;
            let y = (n as f64 * uh / 2.)
                - f64::cos(i as f64 / (n as f64 - 1.) * PI) * (n as f64 * uh / 2.);
            let a = noise(x * 0.005, Some(y * 0.005), None) * PI * 2. - PI;
            let r = noise(x * 0.005, Some(y * 0.005), None);
            let dx = f64::cos(a) * r * noise_x;
            let dy = f64::cos(a) * r * noise_y;
            pts.push((x + dx, y + dy));
        }
    }
    let mut out = vec![];

    let mut whs = vec![];
    for i in 0..n {
        for j in 0..m {
            if i == 0 || j == 0 || i == n - 1 || j == m - 1 {
                whs.push((uw / 2., uh / 2.));
                continue;
            }
            let a = pts[i * m + j];
            let b = pts[i * m + j + 1];
            let c = pts[i * m + j - 1];
            let d = pts[(i - 1) * m + j];
            let e = pts[(i + 1) * m + j];

            let dw = (dist(a, b) + dist(a, c)) / 4.;
            let dh = (dist(a, d) + dist(a, e)) / 4.;
            whs.push((dw, dh));
        }
    }
    let mut j = 1;
    while j < m as i64 - 1 {
        for i in 1..n - 1 {
            let (x, y) = pts[i * m + j as usize];
            let (dw, dh) = whs[i * m + j as usize];
            let q = trsl_poly(&squama_mask(dw, dh), x, y);
            let p = squama_func((x, y), (dw, dh));
            let p: Vec<Polyline> = p.into_iter().map(|a| trsl_poly(&a, x, y)).collect();
            if !interclip {
                out.extend(p.into_iter());
            } else {
                if let Some(c) = clipper {
                    out.extend(clip_multi(&p, &c).dont_clip.into_iter());
                    clipper = Some(poly_union(&c, &q, None));
                } else {
                    out.extend(p.into_iter());
                    clipper = Some(q);
                }
            }
        }
        for i in 1..n - 1 {
            let a = pts[i * m + j as usize];
            let b = pts[i * m + j as usize + 1];
            let c = pts[(i + 1) * m + j as usize];
            let d = pts[(i + 1) * m + j as usize + 1];

            let (dwa, dha) = whs[i * m + j as usize];
            let (dwb, dhb) = whs[i * m + j as usize + 1];
            let (dwc, dhc) = whs[(i + 1) * m + j as usize];
            let (dwd, dhd) = whs[(i + 1) * m + j as usize + 1];

            let (x, y) = ((a.0 + b.0 + c.0 + d.0) / 4., (a.1 + b.1 + c.1 + d.1) / 4.);
            let (mut dw, dh) = ((dwa + dwb + dwc + dwd) / 4., (dha + dhb + dhc + dhd) / 4.);
            dw *= 1.2;
            let q = trsl_poly(&squama_mask(dw, dh), x, y);

            let p: Vec<Polyline> = squama_func((x, y), (dw, dh))
                .into_iter()
                .map(|a| trsl_poly(&a, x, y))
                .collect();
            if !interclip {
                out.extend(p.into_iter());
            } else {
                if let Some(c) = clipper {
                    out.extend(clip_multi(&p, &c).dont_clip.into_iter());
                    clipper = Some(poly_union(&c, &q, None));
                } else {
                    out.extend(p.into_iter());
                    clipper = Some(q);
                }
            }
        }
        j += 1;
    }
    // for i in 0..n-1 {
    //   for j in 0..m-1; j++){
    //     let a= pts[i*m+j];
    //     let b= pts[i*m+j+1];
    //     let c = pts[(i+1)*m+j];
    //     out.push([a,b]);
    //     out.push([a,c]);
    //   }
    // }
    return out;
}

fn fish_body_a(
    curve0: &Polyline,
    curve1: &Polyline,
    scale_scale: f64,
    pattern_func: Option<Rc<dyn Fn(Point) -> bool>>,
) -> Vec<Polyline> {
    let mut curve2 = vec![];
    let mut curve3 = vec![];
    for i in 0..curve0.len() {
        curve2.push(lerp2d(curve0[i], curve1[i], 0.95));
        curve3.push(lerp2d(curve0[i], curve1[i], 0.85));
    }

    let outline1 = curve0.concat(&curve1.rev());
    let outline2 = curve0.concat(&curve2.rev());
    let outline3 = curve0.concat(&curve3.rev());

    let bbox = get_boundingbox(&curve0.concat(curve1));
    let m = (bbox.w / (scale_scale * 15.)).trunc() as usize;
    let n = (bbox.h / (scale_scale * 15.)).trunc() as usize;
    let uw = bbox.w / m as f64;
    let uh = bbox.h / n as f64;
    let funky: Box<dyn Fn(Point, Point) -> Vec<Polyline>> = if let Some(f) = pattern_func {
        Box::new(move |(x, y), (w, h)| squama(w, h, Some(f((x, y)) as usize * 3)))
    } else {
        Box::new(|(x, y), (w, h)| squama(w, h, None))
    };

    let mut sq = squama_mesh(m, n + 3, (uw, uh), funky, (uw * 3., uh * 3.), Some(true));

    sq = sq
        .into_iter()
        .map(|a| trsl_poly(&a, bbox.x, bbox.y - uh * 1.5))
        .collect();
    let o0 = clip_multi(&sq, &outline2).clip;
    let mut o1 = clip_multi(&o0, &outline3);
    o1.dont_clip = o1.dont_clip.into_iter().filter(|x| rand() < 0.6).collect();

    [curve0.clone(), curve1.rev()]
        .into_iter()
        .chain(o1.clip)
        .chain(o1.dont_clip)
        .collect()
}

pub fn fish_body_b(
    curve0: &Polyline,
    curve1: &Polyline,
    scale_scale: f64,
    pattern_func: Option<Rc<dyn Fn((f64, f64)) -> bool>>,
) -> Vec<Polyline> {
    let mut curve2 = vec![];
    for i in 0..curve0.len() {
        curve2.push(lerp2d(curve0[i], curve1[i], 0.95));
    }
    let outline1 = curve0.concat(&curve1.rev());
    let outline2 = curve0.concat(&curve2.rev());

    let bbox = get_boundingbox(&curve0.concat(curve1));
    let m = (bbox.w / (scale_scale * 5.)).trunc();
    let n = (bbox.h / (scale_scale * 5.)).trunc();
    let uw = bbox.w / m;
    let uh = bbox.h / n;

    let sq = squama_mesh(
        m as usize,
        n as usize + 16,
        (uw, uh),
        Box::new(|(x, y), (w, h)| squama(w * 0.7, h * 0.6, Some(0))),
        (uw * 8., uh * 8.),
        Some(false),
    )
    .iter()
    .map(|a| trsl_poly(a, bbox.x, bbox.y - uh * 8.))
    .collect();
    let o0 = clip_multi(&sq, &outline2).clip;

    let mut o1 = vec![];
    for line in o0 {
        let (x, y) = line[0];
        let t = (y - bbox.y) / bbox.h;
        // if (rand() > t){
        //   o1.push(line);
        // }
        // if ((!!(x/30))%2 || (rand() > t && rand()>t)){
        //   o1.push(line);
        // }
        if let Some(funky) = &pattern_func {
            if funky((x, y)) || (rand() > t && rand() > t) {
                o1.push(line);
            }
        } else {
            if rand() > t {
                o1.push(line);
            }
        }
    }

    [curve0.clone(), curve1.rev()]
        .into_iter()
        .chain(o1)
        .collect()
}

pub fn fish_body_c(curve0: &Polyline, curve1: &Polyline, scale_scale: f64) -> Vec<Polyline> {
    let step = 6. * scale_scale;

    let mut curve2 = vec![];
    let mut curve3 = vec![];

    for i in 0..curve0.len() {
        curve2.push(lerp2d(curve0[i], curve1[i], 0.95));
        curve3.push(lerp2d(curve0[i], curve1[i], 0.4));
    }
    let outline1 = curve0.concat(&curve1.rev());
    let outline2 = curve0.concat(&curve2.rev());

    let mut bbox = get_boundingbox(&curve0.concat(&curve1));
    bbox.x -= step;
    bbox.y -= step;
    bbox.w += step * 2.;
    bbox.h += step * 2.;

    let mut lines = vec![curve3.rev()];

    for i in successors(Some(-bbox.h), |i| {
        let next = i + step;
        (next < bbox.w).then_some(next)
    }) {
        lines.push(vec![
            (bbox.x + i, bbox.y),
            (bbox.x + i + bbox.h, bbox.y + bbox.h),
        ]);
    }

    for i in successors(Some(0.), |i| {
        let next = i + step;
        (next < bbox.w + bbox.h).then_some(next)
    }) {
        lines.push(vec![
            (bbox.x + i, bbox.y),
            (bbox.x + i - bbox.h, bbox.y + bbox.h),
        ]);
    }
    for i in 0..lines.len() {
        lines[i] = resample(&lines[i], 4.);
        for j in 0..lines[i].len() {
            let (x, y) = lines[i][j];
            let t = (y - bbox.y) / bbox.h;
            let y1 = -f64::cos(t * PI) * bbox.h / 2. + bbox.y + bbox.h / 2.;

            let dx = (noise(x * 0.005, Some(y1 * 0.005), Some(0.1)) - 0.5) * 50.;
            let dy = (noise(x * 0.005, Some(y1 * 0.005), Some(1.2)) - 0.5) * 50.;

            lines[i][j].0 += dx;
            lines[i][j].1 = y1 + dy;
        }
    }

    let mut o0 = clip_multi(&lines, &outline2).clip;

    o0 = binclip_multi(
        &o0,
        Rc::new(|(x, y), t| (rand() > t as f64 || rand() > t as f64)),
    )
    .clip;

    [curve0.clone(), curve1.rev()]
        .into_iter()
        .chain(o0.into_iter())
        .collect()
}

pub fn fish_body_d(curve0: &Polyline, curve1: &Polyline, scale_scale: f64) -> Vec<Polyline> {
    let mut curve2 = vec![];
    for i in 0..curve0.len() {
        curve2.push(lerp2d(curve0[i], curve1[i], 0.4));
    }
    let curve0 = resample(curve0, 10. * scale_scale);
    let curve1 = resample(curve1, 10. * scale_scale);
    curve2 = resample(&curve2, 10. * scale_scale);

    let outline1 = curve0.concat(&curve1.rev());
    let outline2 = curve0.concat(&curve2.rev());

    let mut o0 = vec![curve2.clone()];
    for i in 3..curve0.len().min(curve1.len()).min(curve2.len()) {
        o0.push(vec![curve0[i], curve2[i - 3]]);
        o0.push(vec![curve2[i - 3], curve1[i]]);
    }

    let mut o1 = vec![];
    for i in 0..o0.len() {
        o0[i] = resample(&o0[i], 4.);
        for j in 0..o0[i].len() {
            let (x, y) = o0[i][j];
            let dx = 30. * (noise(x * 0.01, Some(y * 0.01), Some(-1.)) - 0.5);
            let dy = 30. * (noise(x * 0.01, Some(y * 0.01), Some(9.)) - 0.5);
            o0[i][j].0 += dx;
            o0[i][j].1 += dy;
        }

        o1.extend(
            binclip(&o0[i], |(x, y), t| {
                (rand() > (t as f64 * PI).cos() && rand() < x / 500.)
                    || (rand() > (t as f64 * PI).cos() && rand() < x / 500.)
            })
            .clip,
        );
    }
    o1 = clip_multi(&o1, &outline1).clip;

    let sh = vein_shape(&outline1, None);

    [curve0, curve1.rev()]
        .into_iter()
        .chain(o1)
        .chain(sh)
        .collect()
}

pub fn fin_a(
    curve: &Polyline,
    ang0: f64,
    ang1: f64,
    func: Box<dyn Fn(f64) -> f64>,
    clip_root_opt: Option<bool>,
    curvature0_opt: Option<f64>,
    curvature1_opt: Option<f64>,
    softness_opt: Option<f64>,
) -> (Polyline, Vec<Polyline>) {
    let clip_root = clip_root_opt.unwrap_or(false);
    let curvature0 = curvature0_opt.unwrap_or(0.);
    let curvature1 = curvature1_opt.unwrap_or(0.);
    let softness = softness_opt.unwrap_or(10.);
    let mut angs = vec![];
    for i in 0..curve.len() {
        if i == 0 {
            angs.push(
                f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0) - PI / 2.,
            );
        } else if i == curve.len() - 1 {
            angs.push(
                f64::atan2(curve[i].1 - curve[i - 1].1, curve[i].0 - curve[i - 1].0) - PI / 2.,
            );
        } else {
            let a0 = f64::atan2(curve[i - 1].1 - curve[i].1, curve[i - 1].0 - curve[i].0);
            let mut a1 = f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0);
            while a1 > a0 {
                a1 -= PI * 2.;
            }
            a1 += PI * 2.;
            let a = (a0 + a1) / 2.;
            angs.push(a);
        }
    }
    let mut out0 = vec![];
    let mut out1 = vec![];
    let mut out2 = vec![];
    let mut out3 = vec![];
    for i in 0..curve.len() {
        let t = i as f64 / (curve.len() - 1) as f64;
        let aa = lerp(ang0, ang1, t);
        let a = angs[i] + aa;
        let w = func(t);

        let (x0, y0) = curve[i];
        let x1 = x0 + f64::cos(a) * w;
        let y1 = y0 + f64::sin(a) * w;

        let mut p = resample(&[(x0, y0), (x1, y1)], 3.);
        for j in 0..p.len() {
            let s = j as f64 / (p.len() - 1) as f64;
            let ss = f64::sqrt(s);
            let (x, y) = p[j];
            let cv = lerp(curvature0, curvature1, t) * f64::sin(s * PI);
            p[j].0 += noise(x * 0.1, Some(y * 0.1), Some(3.)) * ss * softness
                + f64::cos(a - PI / 2.) * cv;
            p[j].1 += noise(x * 0.1, Some(y * 0.1), Some(4.)) * ss * softness
                + f64::sin(a - PI / 2.) * cv;
        }
        if i == 0 {
            out2 = p;
        } else if i == curve.len() - 1 {
            out3 = p.rev();
        } else {
            out0.push(p[p.len() - 1]);
            // if (i % 2){
            let q = &p[(if clip_root { (rand() * 4.) as usize } else { 0 })
                ..(2.max((p.len() as f64 * (rand() * 0.5 + 0.5)) as usize))];
            if !q.is_empty() {
                out1.push(q.iter().map(|x| *x).collect());
            }
            // }
        }
    }
    out0 = resample(&out0, 3.);
    for i in 0..out0.len() {
        let (x, y) = out0[i];
        out0[i].0 += (noise(x * 0.1, Some(y * 0.1), None) * 6. - 3.) * (softness / 10.);
        out0[i].1 += (noise(x * 0.1, Some(y * 0.1), None) * 6. - 3.) * (softness / 10.);
    }
    let o: Polyline = [out2, out0, out3].into_iter().flatten().collect();
    out1.insert(0, o.clone());
    return (o.concat(&curve.rev()), out1);
}

pub fn fin_b(
    curve: &Polyline,
    ang0: f64,
    ang1: f64,
    func: impl Fn(f64) -> f64,
    dark_opt: Option<f64>,
) -> (Polyline, Vec<Polyline>) {
    let dark = dark_opt.unwrap_or(1.);
    let mut angs = vec![];
    for i in 0..curve.len() {
        if (i == 0) {
            angs.push(
                f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0) - (PI / 2.),
            );
        } else if (i == curve.len() - 1) {
            angs.push(
                f64::atan2(curve[i].1 - curve[i - 1].1, curve[i].0 - curve[i - 1].0) - (PI / 2.),
            );
        } else {
            let a0 = f64::atan2(curve[i - 1].1 - curve[i].1, curve[i - 1].0 - curve[i].0);
            let mut a1 = f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0);
            while (a1 > a0) {
                a1 -= PI * 2.;
            }
            a1 += PI * 2.;
            let a = (a0 + a1) / 2.;
            angs.push(a);
        }
    }

    let mut out0 = vec![];
    let mut out1 = vec![];
    let mut out2 = vec![];
    let mut out3 = vec![];
    for i in 0..curve.len() {
        let t = (i as f64 / (curve.len() - 1) as f64);
        let aa = lerp(ang0, ang1, t);
        let a = angs[i] + aa;
        let w = func(t);

        let (x0, y0) = curve[i];
        let x1 = x0 + f64::cos(a) * w;
        let y1 = y0 + f64::sin(a) * w;

        let b = (
            x1 + 0.5 * f64::cos(a - (PI / 2.)),
            y1 + 0.5 * f64::sin(a - (PI / 2.)),
        );
        let c = (
            x1 + 0.5 * f64::cos(a + (PI / 2.)),
            y1 + 0.5 * f64::sin(a + (PI / 2.)),
        );

        let p = (
            curve[i].0 + 1.8 * f64::cos(a - (PI / 2.)),
            curve[i].1 + 1.8 * f64::sin(a - (PI / 2.)),
        );
        let q = (
            curve[i].0 + 1.8 * f64::cos(a + (PI / 2.)),
            curve[i].1 + 1.8 * f64::sin(a + (PI / 2.)),
        );
        out1.push((x1, y1));
        out0.push(vec![p, b, c, q]);
    }

    let n = 10;
    for i in 0..curve.len() - 1 {
        let (a0, q0) = (out0[i][2], out0[i][3]);
        let (p1, a1) = (out0[i + 1][0], out0[i + 1][1]);

        let b = lerp2d(a0, q0, 0.1);
        let c = lerp2d(a1, p1, 0.1);

        let mut o = vec![];
        let ang = f64::atan2(c.1 - b.1, c.0 - b.0);

        for j in 0..n {
            let t = (j as f64 / (n - 1) as f64);
            let d = f64::sin(t * PI) * 2.;
            let a = lerp2d(b, c, t);
            o.push((
                a.0 + f64::cos(ang + (PI / 2.)) * d,
                a.1 + f64::sin(ang + (PI / 2.)) * d,
            ))
        }

        // out2.push([b,c]);
        out2.push(o.clone());

        let m = (f64::min(dist(a0, q0), dist(a1, p1)) / 10. * dark) as u64;
        let e = lerp2d(curve[i], curve[i + 1], 0.5);
        for k in 0..m {
            let mut p = vec![];
            let s = k as f64 / m as f64 * 0.7;
            for j in 1..n - 1 {
                p.push(lerp2d(o[j], e, s));
            }
            out3.push(p);
        }
    }

    let mut out4 = vec![];
    if (out0.len() > 1) {
        let mut clipper = out0[0].clone();
        out4.push(out0[0].clone());
        for i in 1..out0.len() {
            out4.extend(clip(&out0[i], &clipper).dont_clip);
            clipper = poly_union(&clipper, &out0[i], None);
        }
    }

    return (
        out2.clone()
            .into_iter()
            .flatten()
            .chain(curve.clone().into_iter().rev())
            .collect(),
        out4.into_iter().chain(out2).chain(out3).collect(),
    );
}

pub fn finlet(curve: &Polyline, h: f64, dir_opt: Option<i32>) -> (Polyline, Vec<Polyline>) {
    let dir = dir_opt.unwrap_or(1);
    let mut angs = vec![];
    for i in 0..curve.len() {
        if i == 0 {
            angs.push(
                f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0) - PI / 2.,
            );
        } else if i == curve.len() - 1 {
            angs.push(
                f64::atan2(curve[i].1 - curve[i - 1].1, curve[i].0 - curve[i - 1].0) - PI / 2.,
            );
        } else {
            let a0 = f64::atan2(curve[i - 1].1 - curve[i].1, curve[i - 1].0 - curve[i].0);
            let mut a1 = f64::atan2(curve[i + 1].1 - curve[i].1, curve[i + 1].0 - curve[i].0);
            while a1 > a0 {
                a1 -= PI * 2.;
            }
            a1 += PI * 2.;
            let a = (a0 + a1) / 2.;
            angs.push(a);
        }
    }
    let mut out0 = vec![];
    for i in 0..curve.len() {
        let t = i as f64 / (curve.len() - 1) as f64;
        let a = angs[i];
        let mut w = if (i + 1) % 3 != 0 { 0. } else { h };
        if dir > 0 {
            w *= 1. - t * 0.5;
        } else {
            w *= 0.5 + t * 0.5
        }

        let (x0, y0) = curve[i];
        let x1 = x0 + f64::cos(a) * w;
        let y1 = y0 + f64::sin(a) * w;
        out0.push((x1, y1));
    }
    out0 = resample(&out0, 2.);
    for i in 0..out0.len() {
        let (x, y) = out0[i];
        out0[i].0 += noise(x * 0.1, Some(y * 0.1), None) * 2. - 3.;
        out0[i].1 += noise(x * 0.1, Some(y * 0.1), None) * 2. - 3.;
    }
    out0.push(curve[curve.len() - 1]);
    return (out0.concat(&curve.rev()), vec![out0]);
}

pub fn fin_adipose(curve: &Polyline, dx: f64, dy: f64, r: f64) -> (Polyline, Vec<Polyline>) {
    let n = 20;
    let (x0, y0) = curve[curve.len() / 2];
    let (x, y) = (x0 + dx, y0 + dy);
    let (x1, y1) = curve[0];
    let (x2, y2) = curve[curve.len() - 1];
    let d1 = dist((x, y), (x1, y1));
    let d2 = dist((x, y), (x2, y2));
    let a1 = f64::acos(r / d1);
    let a2 = f64::acos(r / d2);
    let a01 = f64::atan2(y1 - y, x1 - x) + a1;
    let mut a02 = f64::atan2(y2 - y, x2 - x) - a2;
    a02 -= PI * 2.;
    while a02 < a01 {
        a02 += PI * 2.;
    }
    let mut out0 = vec![(x1, y1)];
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let a = lerp(a01, a02, t);
        let p = (x + f64::cos(a) * r, y + f64::sin(a) * r);
        out0.push(p);
    }
    out0.push((x2, y2));
    out0 = resample(&out0, 3.);
    for i in 0..out0.len() {
        let t = i as f64 / (out0.len() - 1) as f64;
        let s = f64::sin(t * PI);
        let (x, y) = out0[i];
        out0[i].0 += (noise(x * 0.01, Some(y * 0.01), None) - 0.5) * s * 50.;
        out0[i].1 += (noise(x * 0.01, Some(y * 0.01), None) - 0.5) * s * 50.;
    }
    let cc = out0.concat(&curve.rev());
    let mut out1 = clip(&trsl_poly(&out0, 0., 4.), &cc).clip;
    fn shape((x, y): Point, t: usize) -> bool {
        rand() < (t as f64 * PI).sin()
    }
    out1 = binclip_multi(&out1, Rc::new(shape)).clip;
    return (cc, vec![out0].into_iter().chain(out1).collect());
}

pub fn fish_lip((mut x0, mut y0): Point, (mut x1, mut y1): Point, w: f64) -> Polyline {
    x0 += rand() * 0.001 - 0.0005;
    y0 += rand() * 0.001 - 0.0005;
    x1 += rand() * 0.001 - 0.0005;
    y1 += rand() * 0.001 - 0.0005;
    let h = dist((x0, y0), (x1, y1));
    let a0 = f64::atan2(y1 - y0, x1 - x0);
    let n = 10;
    let ang = f64::acos(w / h);
    let dx = f64::cos(a0 + PI / 2.) * 0.5;
    let dy = f64::sin(a0 + PI / 2.) * 0.5;
    let mut o = vec![(x0 - dx, y0 - dy)];
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let a = lerp(ang, PI * 2. - ang, t) + a0;
        let x = -f64::cos(a) * w + x1;
        let y = -f64::sin(a) * w + y1;
        o.push((x, y));
    }
    o.push((x0 + dx, y0 + dy));
    o = resample(&o, 2.5);
    for i in 0..o.len() {
        let (x, y) = o[i];
        o[i].0 += noise(x * 0.05, Some(y * 0.05), Some(-1.)) * 2. - 1.;
        o[i].1 += noise(x * 0.05, Some(y * 0.05), Some(-2.)) * 2. - 1.;
    }
    return o;
}

pub fn fish_teeth(
    (x0, y0): Point,
    (x1, y1): Point,
    h: f64,
    dir: i64,
    sep_opt: Option<f64>,
) -> Vec<Polyline> {
    let sep = sep_opt.unwrap_or(3.5);
    let n = f64::max(2., dist((x0, y0), (x1, y1)).trunc() / sep) as i64;
    let ang = f64::atan2(y1 - y0, x1 - x0);
    let mut out = vec![];
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let a = lerp2d((x0, y0), (x1, y1), t);
        let w = h * t;
        let b = (
            a.0 + f64::cos(ang + dir as f64 * PI / 2.) * w,
            a.1 + f64::sin(ang + dir as f64 * PI / 2.) * w,
        );
        let c = (a.0 + 1. * f64::cos(ang), a.1 + 1. * f64::sin(ang));
        let d = (a.0 + 1. * f64::cos(ang + PI), a.1 + 1. * f64::sin(ang + PI));
        let e = lerp2d(c, b, 0.7);
        let f = lerp2d(d, b, 0.7);
        let g = (
            a.0 + f64::cos(ang + dir as f64 * (PI / 2. + 0.15)) * w,
            a.1 + f64::sin(ang + dir as f64 * (PI / 2. + 0.15)) * w,
        );
        out.push(vec![c, e, g, f, d])
        // out.push(barbel(...a,10,ang+dir*PI/2.))
    }
    return out;
}

pub fn fish_jaw((x0, y0): Point, (x1, y1): Point, (x2, y2): Point) -> (Polyline, Vec<Polyline>) {
    let n = 10;
    let ang = f64::atan2(y2 - y0, x2 - x0);
    let d = dist((x0, y0), (x2, y2));
    let mut o = vec![];
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let s = f64::sin(t * PI);
        let w = s * d / 20.;
        let p = lerp2d((x2, y2), (x0, y0), t);
        let q = (
            p.0 + f64::cos(ang - PI / 2.) * w,
            p.1 + f64::sin(ang - PI / 2.) * w,
        );
        let qq = (
            q.0 + (noise(q.0 * 0.01, Some(q.1 * 0.01), Some(1.)) - 0.5) * 4. * s,
            q.1 + (noise(q.0 * 0.01, Some(q.1 * 0.01), Some(4.)) - 0.5) * 4. * s,
        );
        o.push(qq);
    }
    return (
        vec![(x2, y2), (x1, y1), (x0, y0)],
        [o.clone()]
            .into_iter()
            .chain(vein_shape(&o, Some(5)))
            .collect(),
    );
}

pub fn fish_eye_a(ex: f64, ey: f64, rad: f64) -> (Polyline, Vec<Polyline>) {
    let n = 20;
    let mut eye0 = vec![];
    let mut eye1 = vec![];
    let mut eye2 = vec![];
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let a = t * PI * 2. + PI / 4. * 3.;
        eye0.push((ex + f64::cos(a) * rad, ey + f64::sin(a) * rad));
        if t > 0.5 {
            eye1.push((
                ex + f64::cos(a) * (rad * 0.8),
                ey + f64::sin(a) * (rad * 0.8),
            ));
        }
        eye2.push((
            ex + f64::cos(a) * (rad * 0.4) - 0.75,
            ey + f64::sin(a) * (rad * 0.4) - 0.75,
        ));
    }

    let ef = shade_shape(&eye2, Some(2.7), Some(10.), Some(10.));
    return (
        eye0.clone(),
        [eye0, eye1, eye2].into_iter().chain(ef).collect(),
    );
}

pub fn fish_eye_b(ex: f64, ey: f64, rad: f64) -> (Polyline, Vec<Polyline>) {
    let n = 20;
    let mut eye0 = vec![];
    let mut eye1 = vec![];
    let mut eye2 = vec![];
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let a = t * PI * 2. + E;
        eye0.push((ex + f64::cos(a) * rad, ey + f64::sin(a) * rad));
        eye2.push((
            ex + f64::cos(a) * (rad * 0.4),
            ey + f64::sin(a) * (rad * 0.4),
        ));
    }
    let m = ((rad * 0.6) / 2.).trunc() as usize;
    for i in 0..m {
        let r = rad - i as f64 * 2.;
        let mut e = vec![];
        for i in 0..n {
            let t = i as f64 / (n as f64 - 1.);
            let a = lerp(PI * 7. / 8., PI * 13. / 8., t);
            e.push((ex + f64::cos(a) * r, ey + f64::sin(a) * r));
        }
        eye1.push(e);
    }
    let mut trig = vec![
        (
            ex + f64::cos(-PI * 3. / 4.) * (rad * 0.9),
            ey + f64::sin(-PI * 3. / 4.) * (rad * 0.9),
        ),
        (ex + 1., ey + 1.),
        (
            ex + f64::cos(-PI * 11. / 12.) * (rad * 0.9),
            ey + f64::sin(-PI * 11. / 12.) * (rad * 0.9),
        ),
    ];
    trig = resample(&trig, 3.);
    for i in 0..trig.len() {
        let (mut x, mut y) = trig[i];
        x += noise(x * 0.1, Some(y * 0.1), Some(22.)) * 4. - 2.;
        y += noise(x * 0.1, Some(y * 0.1), Some(33.)) * 4. - 2.;
        trig[i] = (x, y);
    }

    let mut ef = fill_shape(&eye2, Some(1.5));

    ef = clip_multi(&ef, &trig).dont_clip;
    eye1 = clip_multi(&eye1, &trig).dont_clip;
    let eye2_clip = clip(&eye2, &trig).dont_clip;

    return (
        eye0.clone(),
        vec![eye0]
            .into_iter()
            .chain(eye1)
            .chain(eye2_clip)
            .chain(ef)
            .collect(),
    );
}

pub fn barbel((mut x, mut y): Point, n: usize, mut ang: f64, dd_opt: Option<f64>) -> Polyline {
    let dd = dd_opt.unwrap_or(3.);
    let mut curve = vec![(x, y)];
    let sd = rand() * PI * 2.;
    let mut ar = 1.;
    for i in 0..n {
        x += f64::cos(ang) * dd;
        y += f64::sin(ang) * dd;
        ang += (noise(i as f64 * 0.1, Some(sd), None) - 0.5) * ar;
        if i < n / 2 {
            ar *= 1.02;
        } else {
            ar *= 0.92;
        }
        curve.push((x, y));
    }
    let mut o0 = vec![];
    let mut o1 = vec![];
    for i in 0..n - 1 {
        let t = i as f64 / (n - 1) as f64;
        let w = 1.5 * (1. - t);

        let b = curve[i];
        let c = curve[i + 1];

        let mut a1 = f64::atan2(c.1 - b.1, c.0 - b.0);
        let a2;

        if let Some(a) = if i == 0 { None } else { curve.get(i - 1) } {
            let a0 = f64::atan2(a.1 - b.1, a.0 - b.0);

            a1 -= PI * 2.;
            while a1 < a0 {
                a1 += PI * 2.;
            }
            a2 = (a0 + a1) / 2.;
        } else {
            a2 = a1 - PI / 2.;
        }

        o0.push((b.0 + f64::cos(a2) * w, b.1 + f64::sin(a2) * w));
        o1.push((b.0 + f64::cos(a2 + PI) * w, b.1 + f64::sin(a2 + PI) * w));
    }
    o0.push(curve[curve.len() - 1]);
    o1.reverse();
    o0.extend(o1);
    return o0;
}

pub fn fish_head(
    (x0, y0): Point,
    (x1, y1): Point,
    (x2, y2): Point,
    mut arg: Params,
) -> (Polyline, Vec<Polyline>) {
    let n = 20;
    let mut curve0 = vec![];
    let mut curve1 = vec![];
    let mut curve2 = vec![];
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let a = PI / 2. * t;
        let x = x1 - pow(f64::cos(a), 1.5) * (x1 - x0);
        let y = y0 - pow(f64::sin(a), 1.5) * (y0 - y1);
        // let x = lerp(x0,x1,t);
        // let y = lerp(y0,y1,t);

        let dx = (noise(x * 0.01, Some(y * 0.01), Some(9.)) * 40. - 20.) * (1.01 - t);
        let dy = (noise(x * 0.01, Some(y * 0.01), Some(8.)) * 40. - 20.) * (1.01 - t);
        curve0.push((x + dx, y + dy));
    }
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.);
        let a = PI / 2. * t;
        let x = x2 - pow(f64::cos(a), 0.8) * (x2 - x0);
        let y = y0 + pow(f64::sin(a), 1.5) * (y2 - y0);

        let dx = (noise(x * 0.01, Some(y * 0.01), Some(9.)) * 40. - 20.) * (1.01 - t);
        let dy = (noise(x * 0.01, Some(y * 0.01), Some(8.)) * 40. - 20.) * (1.01 - t);
        curve1.insert(0, (x + dx, y + dy));
    }
    let ang = f64::atan2(y2 - y1, x2 - x1);
    for i in 1..n - 1 {
        let t = i as f64 / (n as f64 - 1.);
        let p = lerp2d((x1, y1), (x2, y2), t);
        let s = pow(f64::sin(t * PI), 0.5);
        let r = noise(t * 2., Some(1.2), None) * s * 20.;

        let dx = f64::cos(ang - PI / 2.) * r;
        let dy = f64::sin(ang - PI / 2.) * r;
        curve2.push((p.0 + dx, p.1 + dy));
    }
    let outline = curve0
        .iter()
        .chain(curve2.iter())
        .chain(curve1.iter())
        .map(|p| *p)
        .collect();

    let mut inline: Polyline = curve2[(curve2.len() / 3)..]
        .iter()
        .chain(curve1[0..curve1.len() / 2].iter())
        .take(curve0.len())
        .map(|p| *p)
        .collect();
    for i in 0..inline.len() {
        let t = i as f64 / (inline.len() - 1) as f64;
        let s = f64::sin(t * PI).powi(2) * 0.1 + 0.12;
        inline[i] = lerp2d(inline[i], curve0[i], s);
    }
    let dix = (x0 - inline[inline.len() - 1].0) * 0.3;
    let diy = (y0 - inline[inline.len() - 1].1) * 0.2;
    for i in 0..inline.len() {
        inline[i] = (inline[i].0 + dix, inline[i].1 + diy);
    }

    let par = (0.475, 0.375);
    let mut ex = x0 * par.0 + x1 * par.1 + x2 * (1. - par.0 - par.1);
    let mut ey = y0 * par.0 + y1 * par.1 + y2 * (1. - par.0 - par.1);
    let d0 = pt_seg_dist((ex, ey), (x0, y0), (x1, y1));
    let d1 = pt_seg_dist((ex, ey), (x0, y0), (x2, y2));
    if d0 < arg.eye_size && d1 < arg.eye_size {
        arg.eye_size = f64::min(d0, d1);
    } else if d0 < arg.eye_size {
        let ang = f64::atan2(y1 - y0, x1 - x0) + PI / 2.;
        ex = x0 * 0.5 + x1 * 0.5 + f64::cos(ang) * arg.eye_size;
        ey = y0 * 0.5 + y1 * 0.5 + f64::sin(ang) * arg.eye_size;
    }

    let jaw_pt0 = curve1[18 - arg.mouth_size];
    let jaw_l = dist(jaw_pt0, curve1[18]) * arg.jaw_size;
    let jaw_ang0 = f64::atan2(curve1[18].1 - jaw_pt0.1, curve1[18].0 - jaw_pt0.0);
    let jaw_ang = jaw_ang0 - (arg.has_teeth as f64 * 0.5 + 0.5) * arg.jaw_open as f64 * PI / 4.;
    let jaw_pt1 = (
        jaw_pt0.0 + f64::cos(jaw_ang) * jaw_l,
        jaw_pt0.1 + f64::sin(jaw_ang) * jaw_l,
    );

    let (eye0, mut ef) = if arg.eye_type != 0 {
        fish_eye_b(ex, ey, arg.eye_size)
    } else {
        fish_eye_a(ex, ey, arg.eye_size)
    };
    ef = clip_multi(&ef, &outline).clip;

    let inlines = clip(&inline, &eye0).dont_clip;

    let lip0 = fish_lip(jaw_pt0, curve1[18], 3.);

    let lip1 = fish_lip(jaw_pt0, jaw_pt1, 3.);

    let (jc, mut jaw) = fish_jaw(curve1[15 - arg.mouth_size], jaw_pt0, jaw_pt1);

    jaw = clip_multi(&jaw, &lip1).dont_clip;
    jaw = clip_multi(&jaw, &outline).dont_clip;

    let mut teeth0s = vec![];
    let mut teeth1s = vec![];
    if arg.has_teeth != 0 {
        let teeth0 = fish_teeth(
            jaw_pt0,
            curve1[18],
            arg.teeth_length as f64,
            -1,
            Some(arg.teeth_space),
        );
        let teeth1 = fish_teeth(
            jaw_pt0,
            jaw_pt1,
            arg.teeth_length as f64,
            1,
            Some(arg.teeth_space),
        );

        teeth0s = clip_multi(&teeth0, &lip0).dont_clip;
        teeth1s = clip_multi(&teeth1, &lip1).dont_clip;
    }

    let olines = clip(&outline, &lip0).dont_clip;

    let lip0s = clip(&lip0, &lip1).dont_clip;

    let mut sh = shade_shape(&outline, Some(6.), Some(-6.), Some(-6.));
    sh = clip_multi(&sh, &lip0).dont_clip;
    sh = clip_multi(&sh, &eye0).dont_clip;

    let mut sh2 = vein_shape(&outline, Some(arg.head_texture_amount));

    // let sh2 = patternshade_shape(outline,3,(x,y)=>{
    //   return noise(x*0.1,y*0.1)>0.6;
    // })

    sh2 = clip_multi(&sh2, &lip0).dont_clip;
    sh2 = clip_multi(&sh2, &eye0).dont_clip;

    let mut bbs = vec![];

    let mut lip1s = vec![lip1.clone()];

    if arg.has_moustache != 0 {
        let bb0 = barbel(jaw_pt0, arg.moustache_length, PI * 3. / 4., Some(1.5));
        lip1s = clip(&lip1, &bb0).dont_clip;
        jaw = clip_multi(&jaw, &bb0).dont_clip;
        bbs.push(bb0);
    }

    if arg.has_beard != 0 {
        let jaw_pt;
        if !jaw.is_empty() && !jaw[0].is_empty() {
            jaw_pt = jaw[0][!!(jaw[0].len() / 2)];
        } else {
            jaw_pt = curve1[8];
        }
        let bb1 = trsl_poly(
            &barbel(
                jaw_pt,
                arg.beard_length,
                PI * 0.6 + rand() * 0.4 - 0.2,
                None,
            ),
            rand() * 1. - 0.5,
            rand() * 1. - 0.5,
        );
        let bb2 = trsl_poly(
            &barbel(
                jaw_pt,
                arg.beard_length,
                PI * 0.6 + rand() * 0.4 - 0.2,
                None,
            ),
            rand() * 1. - 0.5,
            rand() * 1. - 0.5,
        );
        let bb3 = trsl_poly(
            &barbel(
                jaw_pt,
                arg.beard_length,
                PI * 0.6 + rand() * 0.4 - 0.2,
                None,
            ),
            rand() * 1. - 0.5,
            rand() * 1. - 0.5,
        );

        let mut bb3c = clip_multi(&vec![bb3], &bb2).dont_clip;
        bb3c = clip_multi(&bb3c, &bb1).dont_clip;
        let bb2c = clip_multi(&vec![bb2], &bb1).dont_clip;
        bbs.push(bb1);
        bbs.extend(bb2c.into_iter().chain(bb3c));
    }

    let mut outline_l = vec![
        (0., 0.),
        (curve0.last().unwrap().0, 0.),
        curve0[curve0.len() - 1],
    ];
    outline_l.extend(curve2);
    outline_l.extend([curve1[0], (curve1[0].0, 300.), (0., 300.)]);

    return (
        outline_l,
        [
            olines, inlines, lip0s, lip1s, ef, sh, sh2, bbs, teeth0s, teeth1s, jaw,
        ]
        .into_iter()
        .flatten()
        .collect(),
    );
}

pub fn bean(x: f64) -> f64 {
    f64::powf(0.25 - f64::powf(x - 0.5, 2.), 0.5) * (2.6 + 2.4 * f64::powf(x, 1.5)) * 0.542
}

fn rev_slice<T: Copy>(v: &Vec<T>, start: usize, end: usize) -> Vec<T> {
    let mut out = vec![];
    for i in (start..end).rev() {
        out.push(v[i]);
    }
    out
}

pub fn fish(arg: Params) -> Vec<Polyline> {
    let n = 32;
    let mut curve0 = vec![];
    let mut curve1 = vec![];
    if arg.body_curve_type == 0 {
        let s = arg.body_curve_amount;
        for i in 0..n {
            let t = i as f64 / ((n as f64) - 1.);

            let x = 225. + (t - 0.5) * arg.body_length;
            let y = 150.
                - ((t * PI).sin() * lerp(0.5, 1., noise(t * 2., Some(1.), None)) * s + (1. - s))
                    * arg.body_height as f64;
            curve0.push((x, y));
        }
        for i in 0..n {
            let t = i as f64 / ((n as f64) - 1.);
            let x = 225. + (t - 0.5) * arg.body_length;
            let y = 150.
                + ((t * PI).sin() * lerp(0.5, 1., noise(t * 2., Some(2.), None)) * s + (1. - s))
                    * arg.body_height;
            curve1.push((x, y));
        }
    } else if arg.body_curve_type == 1 {
        for i in 0..n {
            let t = i as f64 / ((n as f64) - 1.);

            let x = 225. + (t - 0.5) * arg.body_length;
            let y = 150.
                - lerp(
                    1. - arg.body_curve_amount,
                    1.,
                    lerp(0., 1., noise(t * 1.2, Some(1.), None)) * bean(1. - t),
                ) * arg.body_height;
            curve0.push((x, y));
        }
        for i in 0..n {
            let t = i as f64 / ((n as f64) - 1.);
            let x = 225. + (t - 0.5) * arg.body_length;
            let y = 150.
                + lerp(
                    1. - arg.body_curve_amount,
                    1.,
                    lerp(0., 1., noise(t * 1.2, Some(2.), None)) * bean(1. - t),
                ) * arg.body_height;
            curve1.push((x, y));
        }
    }
    let mut outline = curve0.concat(&curve1.rev());
    let mut sh = shade_shape(&outline, Some(8.), Some(-12.), Some(-12.));

    let pattern_func: Option<Rc<dyn Fn((f64, f64)) -> bool>> = match arg.pattern_type {
        0 => None, // none
        1 => {
            // (x,y)=>{
            //   return noise(x*0.1,y*0.1)>0.55;
            // }
            Some(pattern_dot(arg.pattern_scale))
        }
        2 => Some(Rc::new(|(x, y)| {
            (noise(x * 0.1, Some(y * 0.1), None) * f64::max(0.35, (y - 10.) / 280.)) < 0.2
        })),
        3 => Some(Rc::new(move |(x, y)| {
            let dx = noise(x * 0.01, Some(y * 0.01), None) * 30.;
            ((x + dx) / (30. * arg.pattern_scale)).trunc() as i64 % 2 == 1
        })),
        4 => None, //small dot;
        _ => panic!("invalid pattern type: {}", arg.pattern_type),
    };

    let mut bd = match arg.scale_type {
        0 => fish_body_a(&curve0, &curve1, arg.scale_scale, pattern_func.clone()),
        1 => fish_body_b(&curve0, &curve1, arg.scale_scale, pattern_func.clone()),
        2 => fish_body_c(&curve0, &curve1, arg.scale_scale),
        3 => fish_body_d(&curve0, &curve1, arg.scale_scale),
        _ => unreachable!(),
    };

    let f0_func: Box<dyn Fn(f64) -> f64>;
    let f0_a0;
    let f0_a1;
    let f0_cv;
    let dl = arg.dorsal_length;
    if arg.dorsal_type == 0 {
        f0_a0 = 0.2 + deviate(0.05);
        f0_a1 = 0.3 + deviate(0.05);
        f0_cv = 0.;

        f0_func = Box::new(move |t| {
            (0.3 + noise(t * 3., None, None) * 0.7) * dl * f64::sin(t * PI).powf(0.5)
        });
    } else if arg.dorsal_type == 1 {
        f0_a0 = 0.6 + deviate(0.05);
        f0_a1 = 0.3 + deviate(0.05);
        f0_cv = arg.dorsal_length / 8.;
        f0_func = Box::new(move |t| dl * ((f64::powi(t - 1., 2)) * 0.5 + (1. - t) * 0.5));
    } else {
        unreachable!();
    }
    let f0_curve;
    let c0: Polyline;
    let mut f0: Vec<Polyline>;
    if arg.dorsal_texture_type == 0 {
        f0_curve = resample(&curve0[arg.dorsal_start..arg.dorsal_end], 5.);
        (c0, f0) = fin_a(
            &f0_curve,
            f0_a0,
            f0_a1,
            f0_func,
            Some(false),
            Some(f0_cv),
            Some(0.),
            None,
        );
    } else {
        f0_curve = resample(&curve0[arg.dorsal_start..arg.dorsal_end], 15.);
        (c0, f0) = fin_b(&f0_curve, f0_a0, f0_a1, f0_func, None);
    }
    f0 = clip_multi(&f0, &trsl_poly(&outline, 0., 0.001)).dont_clip;

    let mut f1_curve = vec![];
    let f1_func: Box<dyn Fn(f64) -> f64>;
    let f1_a0;
    let f1_a1;
    let f1_soft;
    let f1_cv;
    let f1_pt = lerp2d(curve0[arg.wing_start], curve1[arg.wing_end], arg.wing_y);

    for i in 0..10 {
        let t = i as f64 / 9.;
        let y = lerp(
            f1_pt.1 - arg.wing_width / 2.,
            f1_pt.1 + arg.wing_width / 2.,
            t,
        );
        f1_curve.push((f1_pt.0 /*+ f64::sin(t*PI)*2*/, y));
    }
    if arg.wing_type == 0 {
        f1_a0 = -0.4 + deviate(0.05);
        f1_a1 = 0.4 + deviate(0.05);
        f1_soft = 10.;
        f1_cv = 0.;
        f1_func = Box::new(move |t| {
            (40. + (20. + noise(t * 3., None, None) * 70.) * f64::sin(t * PI).powf(0.5)) / 130.
                * arg.wing_length
        });
    } else {
        f1_a0 = 0. + deviate(0.05);
        f1_a1 = 0.4 + deviate(0.05);
        f1_soft = 5.;
        f1_cv = arg.wing_length / 25.;
        f1_func = Box::new(move |t| (arg.wing_length * (1. - t * 0.95)));
    }

    let c1;
    let mut f1;
    if arg.wing_texture_type == 0 {
        f1_curve = resample(&f1_curve, 1.5);
        (c1, f1) = fin_a(
            &f1_curve,
            f1_a0,
            f1_a1,
            f1_func,
            Some(true),
            Some(f1_cv),
            Some(0.),
            Some(f1_soft),
        );
    } else {
        f1_curve = resample(&f1_curve, 4.);
        (c1, f1) = fin_b(&f1_curve, f1_a0, f1_a1, f1_func, Some(0.3));
    }
    bd = clip_multi(&bd, &c1).dont_clip;

    let f2_curve;
    let f2_func: Box<dyn Fn(f64) -> f64>;
    let f2_a0;
    let f2_a1;
    if arg.pelvic_type == 0 {
        f2_a0 = -0.8 + deviate(0.05);
        f2_a1 = -0.5 + deviate(0.05);
        f2_func = Box::new(move |t| {
            (10. + (15. + noise(t * 3., None, None) * 60.) * f64::sin(t * PI).powf(0.5)) / 85.
                * arg.pelvic_length
        });
    } else {
        f2_a0 = -0.9 + deviate(0.05);
        f2_a1 = -0.3 + deviate(0.05);
        f2_func = Box::new(move |t| (t * 0.5 + 0.5) * arg.pelvic_length);
    }

    let mut f2;
    if arg.pelvic_texture_type == 0 {
        f2_curve = resample(
            &rev_slice(&curve1, arg.pelvic_start, arg.pelvic_end),
            if arg.pelvic_type != 0 { 2. } else { 5. },
        );
        (_, f2) = fin_a(&f2_curve, f2_a0, f2_a1, f2_func, None, None, None, None);
    } else {
        f2_curve = resample(
            &rev_slice(&curve1, arg.pelvic_start, arg.pelvic_end),
            if arg.pelvic_type != 0 { 2. } else { 15. },
        );
        (_, f2) = fin_b(&f2_curve, f2_a0, f2_a1, f2_func, None);
    }
    f2 = clip_multi(&f2, &c1).dont_clip;

    let f3_curve;
    let f3_func: Box<dyn Fn(f64) -> f64>;
    let f3_a0;
    let f3_a1;
    if arg.anal_type == 0 {
        f3_a0 = -0.4 + deviate(0.05);
        f3_a1 = -0.4 + deviate(0.05);
        f3_func = Box::new(move |t| {
            (10. + (10. + noise(t * 3., None, None) * 30.) * f64::sin(t * PI).powf(0.5)) / 50.
                * arg.anal_length
        });
    } else {
        f3_a0 = -0.4 + deviate(0.05);
        f3_a1 = -0.4 + deviate(0.05);
        f3_func = Box::new(move |t| arg.anal_length * (t * t * 0.8 + 0.2));
    }
    let mut f3;
    if arg.anal_texture_type == 0 {
        f3_curve = resample(&rev_slice(&curve1, arg.anal_start, arg.anal_end), 5.);
        (_, f3) = fin_a(&f3_curve, f3_a0, f3_a1, f3_func, None, None, None, None);
    } else {
        f3_curve = resample(&rev_slice(&curve1, arg.anal_start, arg.anal_end), 15.);
        (_, f3) = fin_b(&f3_curve, f3_a0, f3_a1, f3_func, None);
    }
    f3 = clip_multi(&f3, &c1).dont_clip;

    let mut f4_curve;
    let c4;
    let mut f4;
    let f4_r = dist(curve0[curve0.len() - 2], curve1[curve1.len() - 2]);
    let mut f4_n = (f4_r / 1.5).trunc();
    f4_n = f64::max(f64::min(f4_n, 20.), 8.);
    let f4_d = f4_r / f4_n;
    // console.log(f4_n,f4_d);
    if arg.tail_type == 0 {
        f4_curve = vec![curve0[curve0.len() - 1], curve1[curve1.len() - 1]];
        f4_curve = resample(&f4_curve, f4_d);
        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| {
                (75. - (10. + noise(t * 3., None, None) * 10.) * f64::sin(3. * t * PI - PI)) / 75.
                    * arg.tail_length
            }),
            Some(true),
            None,
            None,
            None,
        );
    } else if arg.tail_type == 1 {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| arg.tail_length * (f64::sin(t * PI) * 0.5 + 0.5)),
            Some(true),
            None,
            None,
            None,
        );
    } else if arg.tail_type == 2 {
        f4_curve = vec![curve0[curve0.len() - 1], curve1[curve1.len() - 1]];
        f4_curve = resample(&f4_curve, f4_d * 0.7);
        let cv = arg.tail_length / 8.;
        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| (f64::abs(f64::cos(PI * t)) * 0.8 + 0.2) * arg.tail_length),
            Some(true),
            Some(cv),
            Some(-cv),
            None,
        );
    } else if arg.tail_type == 3 {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);

        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| (1. - f64::sin(t * PI) * 0.3) * arg.tail_length),
            Some(true),
            None,
            None,
            None,
        );
    } else if arg.tail_type == 4 {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| (1. - f64::sin(t * PI) * 0.6) * (1. - t * 0.45) * arg.tail_length),
            Some(true),
            None,
            None,
            None,
        );
    } else if arg.tail_type == 5 {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        (c4, f4) = fin_a(
            &f4_curve,
            -0.6,
            0.6,
            Box::new(move |t| (1. - f64::sin(t * PI).powf(0.4) * 0.55) * arg.tail_length),
            Some(true),
            None,
            None,
            None,
        );
    } else {
        unreachable!();
    }
    // f4 = clip_multi(f4,trsl_poly(outline,-1,0)).dont_clip;
    bd = clip_multi(&bd, &trsl_poly(&c4, 1., 0.)).dont_clip;

    f4 = clip_multi(&f4, &c1).dont_clip;

    let mut f5_curve = vec![];
    let mut c5 = vec![];
    let mut f5 = vec![];
    if arg.finlet_type == 0 {
        //pass
    } else if arg.finlet_type == 1 {
        f5_curve = resample(&curve0[arg.dorsal_end..curve0.len() - 2], 5.);
        (c5, f5) = finlet(&f5_curve, 5., None);
        f5_curve = resample(&rev_slice(&curve1, arg.anal_end, curve1.len() - 2), 5.);
        if f5_curve.len() > 1 {
            (c5, f5) = finlet(&f5_curve, 5., None);
        }
    } else if arg.finlet_type == 2 {
        f5_curve = resample(&curve0[27..30], 5.);
        (c5, f5) = fin_adipose(&f5_curve, 20., -5., 6.);
        outline = poly_union(&outline, &trsl_poly(&c5, 0., -1.), None);
    } else {
        f5_curve = resample(&curve0[arg.dorsal_end + 2..curve0.len() - 3], 5.);
        if f5_curve.len() > 2 {
            (c5, f5) = fin_a(
                &f5_curve,
                0.2,
                0.3,
                Box::new(move |t| {
                    (0.3 + noise(t * 3., None, None) * 0.7)
                        * arg.dorsal_length
                        * 0.6
                        * f64::sin(t * PI).powf(0.5)
                }),
                None,
                None,
                None,
                None,
            );
        }
    }
    let (cf, fh) = if arg.neck_type == 0 {
        fish_head(
            (50. - arg.head_length, 150. + arg.nose_height),
            curve0[6],
            curve1[5],
            arg,
        )
    } else {
        fish_head(
            (50. - arg.head_length, 150. + arg.nose_height),
            curve0[5],
            curve1[6],
            arg,
        )
    };
    bd = clip_multi(&bd, &cf).dont_clip;

    sh = clip_multi(&sh, &cf).dont_clip;
    sh = clip_multi(&sh, &c1).dont_clip;

    f1 = clip_multi(&f1, &cf).dont_clip;

    f0 = clip_multi(&f0, &c1).dont_clip;

    let mut sh2 = vec![];
    if let Some(func) = pattern_func {
        if arg.scale_type > 1 {
            sh2 = patternshade_shape(
                &poly_union(&outline, &trsl_poly(&c0, 0., 3.), None),
                3.5,
                func,
            );
        } else {
            sh2 = patternshade_shape(&c0, 4.5, func);
        }
        sh2 = clip_multi(&sh2, &cf).dont_clip;
        sh2 = clip_multi(&sh2, &c1).dont_clip;
    }

    let mut sh3 = vec![];
    if arg.pattern_type == 4 {
        sh3 = smalldot_shape(
            &poly_union(&outline, &trsl_poly(&c0, 0., 5.), None),
            arg.pattern_scale,
        );
        sh3 = clip_multi(&sh3, &c1).dont_clip;
        sh3 = clip_multi(&sh3, &cf).dont_clip;
    }
    bd.extend(f0);
    bd.extend(f1);
    bd.extend(f2);
    bd.extend(f3);
    bd.extend(f4);
    bd.extend(f5);
    bd.extend(fh);
    bd.extend(sh);
    bd.extend(sh2);
    bd.extend(sh3);
    bd
}

fn put_text(txt: String) -> (f64, Vec<Vec<(f64, f64)>>) {
    let base = 500;
    let mut x = 0.;
    let mut o = vec![];
    for c in txt.chars() {
        let ord = c as i64;
        let idx;
        if 65 <= ord && ord <= 90 {
            idx = base + 1 + (ord - 65);
        } else if 97 <= ord && ord <= 122 {
            idx = base + 101 + (ord - 97);
        } else if ord == 46 {
            idx = 710;
        } else if ord == 32 {
            x += 10.;
            continue;
        } else {
            continue;
        }
        let (xmin, xmax, polylines) = compile_hershey(idx);
        o.extend(polylines.iter().map(|p| trsl_poly(p, x - xmin as f64, 0.)));
        x += (xmax - xmin) as f64;
    }
    return (x, o);
}

pub fn reframe(
    mut polylines: Vec<Polyline>,
    pad_opt: Option<f64>,
    text: Option<String>,
) -> Vec<Polyline> {
    let pad = pad_opt.unwrap_or(20.);

    let w = 500. - pad * 2.;
    let h = (300. - pad * 2.) - (if text.is_some() { 10. } else { 0. });
    let bbox = get_boundingbox(&flat(&polylines));
    let sw = w / bbox.w;
    let sh = h / bbox.h;
    let s = sw.min(sh);
    let px = (w - bbox.w * s) / 2.;
    let py = (h - bbox.h * s) / 2.;
    for i in 0..polylines.len() {
        for j in 0..polylines[i].len() {
            let (mut x, mut y) = polylines[i][j];
            x = (x - bbox.x) * s + px + pad;
            y = (y - bbox.y) * s + py + pad;
            polylines[i][j] = (x, y);
        }
    }
    let (mut tw, tp) = put_text(text.unwrap_or(String::new()));
    tw *= 0.3;
    polylines.extend(
        tp.into_iter()
            .map(|p| scl_poly(&shr_poly(&p, -0.3), 0.3, Some(0.3)))
            .map(|p| trsl_poly(&p, 250. - tw / 2., 300. - pad + 5.)),
    );
    return polylines;
}

pub fn cleanup(mut polylines: Vec<Polyline>) -> Vec<Vec<(f64, f64)>> {
    for i in (polylines.len() - 1)..=0 {
        // polylines[i] = approx_poly_dp(polylines[i], 0.1);
        for j in 0..polylines[i].len() {
            polylines[i][j] = (
                (polylines[i][j].0 * 10000.).trunc() / 10000.,
                (polylines[i][j].1 * 10000.).trunc() / 10000.,
            );
        }
        if polylines[i].len() < 2 {
            polylines.splice(i..1, []);
            continue;
        }
        if polylines[i].len() == 2 {
            if dist(polylines[i][0], polylines[i][1]) < 0.9 {
                polylines.splice(i..1, []);
                continue;
            }
        }
    }
    return polylines;
}
