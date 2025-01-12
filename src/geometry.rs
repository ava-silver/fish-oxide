use crate::custom_rand::{noise, rand};
use std::{
    collections::HashMap,
    f64::consts::{E, PI},
    iter::successors,
    rc::Rc,
};

pub type Point = (f64, f64);
pub type Polyline = Vec<Point>;

pub trait PolylineOps {
    fn rev(&self) -> Polyline;
    fn concat(&self, other: &Polyline) -> Polyline;
}

impl PolylineOps for Polyline {
    fn rev(&self) -> Polyline {
        self.iter().rev().map(|p| *p).collect()
    }

    fn concat(&self, other: &Polyline) -> Polyline {
        let mut out = self.clone();
        out.extend(other.iter());
        out
    }
}

pub fn flat(polylines: &Vec<Polyline>) -> Polyline {
    polylines
        .iter()
        .flat_map(|p| p.iter())
        .map(|p| *p)
        .collect()
}

pub fn dist((x0, y0): Point, (x1, y1): Point) -> f64 {
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1. - t) + b * t
}
pub fn lerp2d((x0, y0): Point, (x1, y1): Point, t: f64) -> Point {
    (x0 * (1. - t) + x1 * t, y0 * (1. - t) + y1 * t)
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

pub fn get_boundingbox(points: &Polyline) -> BoundingBox {
    let mut xmin = f64::INFINITY;
    let mut ymin = f64::INFINITY;
    let mut xmax = -f64::INFINITY;
    let mut ymax = -f64::INFINITY;

    for &(x, y) in points {
        xmin = xmin.min(x);
        ymin = ymin.min(y);
        xmax = xmax.max(x);
        ymax = ymax.max(y);
    }

    BoundingBox {
        x: xmin,
        y: ymin,
        w: xmax - xmin,
        h: ymax - ymin,
    }
}

#[derive(Debug, Clone)]
pub struct Intersection {
    t: f64,
    s: f64,
    side: i64,
    other: Option<usize>,
    xy: Option<Point>,
    jump: Option<bool>,
}

pub fn sort_intersections(intersections: &mut Vec<Intersection>) {
    intersections.sort_by(|a, b| a.t.total_cmp(&b.t));
}

pub fn pt_in_pl((x, y): Point, (x0, y0): Point, (x1, y1): Point) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    (x - x0) * dy - (y - y0) * dx
}

pub fn get_side((x, y): Point, (x0, y0): Point, (x1, y1): Point) -> i64 {
    if pt_in_pl((x, y), (x0, y0), (x1, y1)) < 0. {
        1
    } else {
        -1
    }
}

pub fn seg_isect(
    (p0x, p0y): Point,
    (p1x, p1y): Point,
    (q0x, q0y): Point,
    (q1x, q1y): Point,
    is_ray_opt: Option<bool>,
) -> Option<Intersection> {
    let is_ray = is_ray_opt.unwrap_or(false);
    let d0x = p1x - p0x;
    let d0y = p1y - p0y;
    let d1x = q1x - q0x;
    let d1y = q1y - q0y;
    let vc = d0x * d1y - d0y * d1x;
    if vc == 0. {
        return None;
    }
    let vcn = vc * vc;
    let q0x_p0x = q0x - p0x;
    let q0y_p0y = q0y - p0y;
    let vc_vcn = vc / vcn;
    let t = (q0x_p0x * d1y - q0y_p0y * d1x) * vc_vcn;
    let s = (q0x_p0x * d0y - q0y_p0y * d0x) * vc_vcn;
    if 0. <= t && (is_ray || t < 1.) && 0. <= s && s < 1. {
        return Some(Intersection {
            t,
            s,
            side: get_side((p0x, p0y), (p1x, p1y), (q0x, q0y)),
            other: None,
            xy: Some((p1x * t + p0x * (1. - t), p1y * t + p0y * (1. - t))),
            jump: None,
        });
    }
    return None;
}

pub fn poly_bridge(poly0: &Polyline, poly1: &Polyline) -> Polyline {
    let mut dmin = f64::INFINITY;
    let mut imin = (0, 0);
    for i in 0..poly0.len() {
        for j in 0..poly1.len() {
            let (x0, y0) = poly0[i];
            let (x1, y1) = poly1[j];
            let dx = x0 - x1;
            let dy = y0 - y1;
            let d2 = dx * dx + dy * dy;
            if d2 < dmin {
                dmin = d2;
                imin = (i, j);
            }
        }
    }
    poly0[0..imin.0]
        .iter()
        .chain(poly1[imin.1..poly1.len()].iter())
        .chain(poly1[0..imin.1].iter())
        .chain(poly0[imin.0..poly0.len()].iter())
        .map(|p| *p)
        .collect()
}
#[derive(Debug, Clone)]
pub struct Vertex {
    xy: Point,
    isects: Vec<Intersection>,
    isects_map: HashMap<(usize, usize), Intersection>,
}

pub fn build_vertices(
    poly: &Polyline,
    other: &Polyline,
    out: &mut Vec<Vertex>,
    oout: &mut Vec<Vertex>,
    idx: usize,
    self_isect: bool,
    has_isect: &mut bool,
) {
    let n = poly.len();
    let m = other.len();
    if self_isect {
        for i in 0..n {
            let id = (idx, i);
            let i1 = (i + 1 + n) % n;
            let a = poly[i];
            let b = poly[i1];
            for j in 0..n {
                let jd = (idx, j);
                let j1 = (j + 1 + n) % n;
                if i == j || i == j1 || i1 == j || i1 == j1 {
                    continue;
                }
                let c = poly[j];
                let d = poly[j1];
                let xx_opt = if let Some(ox) = out[j].isects_map.get(&id) {
                    Some(Intersection {
                        t: ox.s,
                        s: ox.t,
                        xy: ox.xy,
                        other: None,
                        side: get_side(a, b, c),
                        jump: None,
                    })
                } else {
                    seg_isect(a, b, c, d, None)
                };
                if let Some(mut xx) = xx_opt {
                    xx.other = Some(j);
                    xx.jump = Some(false);
                    let p = out.get_mut(i).unwrap();
                    p.isects.push(xx.clone());
                    p.isects_map.insert(jd, xx);
                }
            }
        }
    }

    for i in 0..n {
        let id = (idx, i);
        let p = out.get_mut(i).unwrap();
        let i1 = (i + 1 + n) % n;
        let a = poly[i];
        let b = poly[i1];
        for j in 0..m {
            let jd = (1 - idx, j);
            let j1 = (j + 1 + m) % m;
            let c = other[j];
            let d = other[j1];
            let xx_opt = if let Some(ox) = oout[j].isects_map.get(&id) {
                Some(Intersection {
                    t: ox.s,
                    s: ox.t,
                    xy: ox.xy,
                    other: None,
                    side: get_side(a, b, c),
                    jump: None,
                })
            } else {
                seg_isect(a, b, c, d, None)
            };
            if let Some(mut xx) = xx_opt {
                *has_isect = true;
                xx.other = Some(j);
                xx.jump = Some(true);
                p.isects.push(xx.clone());
                p.isects_map.insert(jd, xx);
            }
        }
        sort_intersections(&mut p.isects);
    }
}

pub fn poly_union(poly0: &Polyline, poly1: &Polyline, self_isect_opt: Option<bool>) -> Polyline {
    let self_isect = self_isect_opt.unwrap_or(false);
    let mut verts0 = poly0
        .iter()
        .map(|&xy| Vertex {
            xy,
            isects: Vec::new(),
            isects_map: HashMap::new(),
        })
        .collect();
    let mut verts1 = poly1
        .iter()
        .map(|&xy| Vertex {
            xy,
            isects: Vec::new(),
            isects_map: HashMap::new(),
        })
        .collect();

    let mut has_isect = false;

    build_vertices(
        poly0,
        poly1,
        &mut verts0,
        &mut verts1,
        0,
        self_isect,
        &mut has_isect,
    );
    build_vertices(
        poly1,
        poly0,
        &mut verts1,
        &mut verts0,
        1,
        self_isect,
        &mut has_isect,
    );

    if !has_isect {
        if !self_isect {
            return poly_bridge(poly0, poly1);
        } else {
            return poly_union(&poly_bridge(poly0, poly1), &Vec::new(), Some(true));
        }
    }

    let mut isect_mir = HashMap::new();
    pub fn mirror_isects(
        verts0: &mut Vec<Vertex>,
        verts1: &mut Vec<Vertex>,
        idx: usize,
        isect_mir: &mut HashMap<(usize, usize, i64), (usize, usize, i64)>,
    ) {
        let n = verts0.len();
        for i in 0..n {
            let m = verts0[i].isects.len();
            for j in 0..m {
                let id = (idx, i, j as i64);
                let jump = verts0[i].isects[j].jump.unwrap_or(false);
                let jd = if jump { 1 - idx } else { idx };
                let k = verts0[i].isects[j].other.unwrap();
                let z = (if jump { &verts1 } else { &verts0 })[k]
                    .isects
                    .iter()
                    .position(|x| (x.jump == Some(jump) && x.other == Some(i)))
                    .unwrap();
                isect_mir.insert(id, (jd, k, z as i64));
            }
        }
    }
    mirror_isects(&mut verts0, &mut verts1, 0, &mut isect_mir);
    mirror_isects(&mut verts1, &mut verts0, 1, &mut isect_mir);

    pub fn trace_outline(
        idx: usize,
        i0: usize,
        j0: i64,
        dir: i64,
        verts0: &Vec<Vertex>,
        verts1: &Vec<Vertex>,
        isect_mir: &HashMap<(usize, usize, i64), (usize, usize, i64)>,
    ) -> Option<Polyline> {
        pub fn trace_from(
            mut zero: Option<(usize, usize, i64)>,
            verts0: &Vec<Vertex>,
            verts1: &Vec<Vertex>,
            idx: usize,
            i0: usize,
            j0: i64,
            dir: i64,
            out: &mut Polyline,
            isect_mir: &HashMap<(usize, usize, i64), (usize, usize, i64)>,
        ) -> bool {
            if zero == None {
                zero = Some((idx, i0, j0));
            } else if idx == zero.unwrap().0 && i0 == zero.unwrap().1 && j0 == zero.unwrap().2 {
                return true;
            }
            let verts = if idx > 0 { verts1 } else { verts0 };
            let n = verts.len();
            let p = &verts[i0];
            let i1 = (((i0 + n) as i64) + dir) as usize % n;
            if j0 == -1 {
                out.push(p.xy);
                if dir < 0 {
                    return trace_from(
                        zero,
                        verts0,
                        verts1,
                        idx,
                        i1,
                        verts[i1].isects.len() as i64 - 1,
                        dir,
                        out,
                        isect_mir,
                    );
                } else if verts[i0].isects.is_empty() {
                    return trace_from(zero, verts0, verts1, idx, i1, -1, dir, out, isect_mir);
                } else {
                    return trace_from(zero, verts0, verts1, idx, i0, 0, dir, out, isect_mir);
                }
            } else if j0 >= p.isects.len() as i64 {
                return trace_from(zero, verts0, verts1, idx, i1, -1, dir, out, isect_mir);
            } else {
                out.push(p.isects[j0 as usize].xy.unwrap());

                let q = &p.isects[j0 as usize];
                let (jdx, k, z) = isect_mir[&(idx, i0, j0)];
                if q.side * dir < 0 {
                    return trace_from(
                        zero,
                        verts0,
                        verts1,
                        jdx,
                        k,
                        (z - 1) as i64,
                        -1,
                        out,
                        isect_mir,
                    );
                }
                return trace_from(
                    zero,
                    verts0,
                    verts1,
                    jdx,
                    k,
                    (z + 1) as i64,
                    1,
                    out,
                    isect_mir,
                );
            }
        }
        let zero = None;
        let mut out = Vec::new();
        let success = trace_from(zero, verts0, verts1, idx, i0, j0, dir, &mut out, isect_mir);
        if !success || out.len() < 3 {
            return None;
        }
        Some(out)
    }

    let mut xmin = f64::INFINITY;
    let mut amin = (0, 0);
    for i in 0..poly0.len() {
        if poly0[i].0 < xmin {
            xmin = poly0[i].0;
            amin = (0, i);
        }
    }
    for i in 0..poly1.len() {
        if poly1[i].0 < xmin {
            xmin = poly1[i].0;
            amin = (1, i);
        }
    }

    pub fn check_concavity(poly: &Polyline, idx: usize) -> i64 {
        let n = poly.len();
        let a = poly[(idx as i32 - 1 + n as i32) as usize % n];
        let b = poly[idx];
        let c = poly[(idx + 1) % n];
        let cw = get_side(a, b, c);
        return cw;
    }

    let cw = check_concavity(if amin.0 != 0 { &poly1 } else { &poly0 }, amin.1);
    let outline = trace_outline(amin.0, amin.1, -1, cw, &verts0, &verts1, &isect_mir);
    outline.unwrap_or_else(|| Vec::new())
}

pub fn seg_isect_poly(
    p0: Point,
    p1: Point,
    poly: &Polyline,
    is_ray_opt: Option<bool>,
) -> Vec<Intersection> {
    let is_ray = is_ray_opt.unwrap_or(false);
    let n = poly.len();
    let mut isects = Vec::new();
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % n];

        if let Some(xx) = seg_isect(p0, p1, a, b, Some(is_ray)) {
            isects.push(xx);
        }
    }
    sort_intersections(&mut isects);
    isects
}

#[derive(Default)]
pub struct ClipSegments {
    pub clip: Vec<Polyline>,
    pub dont_clip: Vec<Polyline>,
}

impl ClipSegments {
    pub fn new_with_empty() -> Self {
        ClipSegments {
            clip: vec![vec![]],
            dont_clip: vec![vec![]],
        }
    }
    pub fn extend(&mut self, other: Self) {
        self.clip.extend(other.clip.into_iter());
        self.dont_clip.extend(other.dont_clip.into_iter());
    }
    pub fn filter_empty(mut self) -> Self {
        self.clip = self.clip.into_iter().filter(|l| !l.is_empty()).collect();
        self.dont_clip = self
            .dont_clip
            .into_iter()
            .filter(|l| !l.is_empty())
            .collect();
        self
    }
    pub fn get(&self, clip: bool) -> &Vec<Polyline> {
        if clip {
            &self.clip
        } else {
            &self.dont_clip
        }
    }

    pub fn get_mut(&mut self, clip: bool) -> &mut Vec<Polyline> {
        if clip {
            &mut self.clip
        } else {
            &mut self.dont_clip
        }
    }
}

pub fn clip(polyline: &Polyline, polygon: &Polyline) -> ClipSegments {
    if polyline.is_empty() {
        return ClipSegments::default();
    }
    let zero = seg_isect_poly(
        polyline[0],
        (polyline[0].0 + E, polyline[0].1 + PI),
        polygon,
        Some(true),
    )
    .len()
        % 2
        != 0;
    let mut out = ClipSegments::new_with_empty();
    let mut io = zero;
    for i in 0..polyline.len() {
        let a = polyline[i];
        let b_opt = polyline.get(i + 1);
        let idx = out.get(io).len() - 1;
        out.get_mut(io)[idx].push(a);
        let Some(&b) = b_opt else {
            break;
        };

        let isects = seg_isect_poly(a, b, polygon, Some(false));
        for j in 0..isects.len() {
            let idx = out.get(io).len() - 1;
            out.get_mut(io)[idx].push(isects[j].xy.unwrap());
            io = !io;
            out.get_mut(io).push(vec![isects[j].xy.unwrap()]);
        }
    }
    out.filter_empty()
}

pub fn clip_multi(polylines: &Vec<Polyline>, polygon: &Polyline) -> ClipSegments {
    let mut out = ClipSegments::default();
    for polyline in polylines {
        out.extend(clip(polyline, polygon));
    }
    return out;
}

pub fn binclip(polyline: &Polyline, func: impl Fn(Point, usize) -> bool) -> ClipSegments {
    if polyline.is_empty() {
        return ClipSegments::default();
    }
    let mut bins = Vec::new();
    for i in 0..polyline.len() {
        let t = i / (polyline.len() - 1);
        bins.push(func(polyline[i], t));
    }
    let zero = bins[0];
    let mut out = ClipSegments::new_with_empty();
    let mut io = zero;
    for i in 0..polyline.len() {
        let a = polyline[i];
        let b_opt = polyline.get(i + 1);
        let idx = out.get(io).len() - 1;
        out.get_mut(io)[idx].push(a);
        let Some(&b) = b_opt else {
            break;
        };

        let do_isect = bins[i] != bins[i + 1];

        if do_isect {
            let pt = lerp2d(a, b, 0.5);
            let idx = out.get(io).len() - 1;
            out.get_mut(io)[idx].push(pt);
            io = !io;
            out.get_mut(io).push(vec![pt]);
        }
    }
    out.filter_empty()
}

pub fn binclip_multi(
    polylines: &Vec<Polyline>,
    f: Rc<dyn Fn(Point, usize) -> bool>,
) -> ClipSegments {
    let mut out = ClipSegments::default();
    for polyline in polylines {
        out.extend(binclip(polyline, f.as_ref()));
    }
    return out;
}

pub fn trsl_poly(poly: &Polyline, x: f64, y: f64) -> Polyline {
    return poly.iter().map(|(x0, y0)| (x0 + x, y0 + y)).collect();
}

pub fn shade_shape(
    poly: &Polyline,
    step_opt: Option<f64>,
    dx_opt: Option<f64>,
    dy_opt: Option<f64>,
) -> Vec<Polyline> {
    let step = step_opt.unwrap_or(5.);
    let dx = dx_opt.unwrap_or(10.);
    let dy = dy_opt.unwrap_or(20.);
    let mut bbox = get_boundingbox(poly);
    bbox.x -= step;
    bbox.y -= step;
    bbox.w += step * 2.;
    bbox.h += step * 2.;
    let mut lines: Vec<_> = successors(Some(-bbox.h), |i| {
        let next = i + step;
        (next < bbox.w).then_some(next)
    })
    .map(|i| vec![(bbox.x + i, bbox.y), (bbox.x + i + bbox.h, bbox.y + bbox.h)])
    .collect();

    lines = clip_multi(&lines, poly).clip;

    let carve = trsl_poly(poly, -dx, -dy);

    lines = clip_multi(&lines, &carve).dont_clip;

    for i in 0..lines.len() {
        let line = &lines[i];
        let mut a = line[0];
        let mut b = line[1];
        let s = (rand()) * 0.5;
        if dy > 0. {
            a = lerp2d(a, b, s);
            lines[i][0] = a;
        } else {
            b = lerp2d(b, a, s);
            lines[i][1] = b;
        }
    }

    lines
}

pub fn fill_shape(poly: &Polyline, step_opt: Option<f64>) -> Vec<Vec<(f64, f64)>> {
    let step = step_opt.unwrap_or(5.);
    let mut bbox = get_boundingbox(poly);
    bbox.x -= step as f64;
    bbox.y -= step as f64;
    bbox.w += step as f64 * 2.;
    bbox.h += step as f64 * 2.;
    let mut lines = vec![];

    for i in successors(Some(0.), |i| {
        let next = i + step;
        (next < bbox.w + bbox.h / 2.).then_some(next)
    }) {
        let x0 = bbox.x + i;
        let y0 = bbox.y;
        let x1 = bbox.x + i - bbox.h / 2.;
        let y1 = bbox.y + bbox.h;
        lines.push(vec![(x0, y0), (x1, y1)]);
    }
    lines = clip_multi(&lines, &poly).clip;

    return lines;
}

pub fn patternshade_shape(
    poly: &Polyline,
    step: f64,
    pattern_func: Rc<dyn Fn((f64, f64)) -> bool>,
) -> Vec<Polyline> {
    let mut bbox = get_boundingbox(poly);
    bbox.x -= step;
    bbox.y -= step;
    bbox.w += step * 2.;
    bbox.h += step * 2.;
    let mut lines = vec![];
    for i in successors(Some(-bbox.h / 2.), |i| {
        let next = i + step;
        (next < bbox.w).then_some(next)
    }) {
        let x0 = bbox.x + i;
        let y0 = bbox.y;
        let x1 = bbox.x + i + bbox.h / 2.;
        let y1 = bbox.y + bbox.h;
        lines.push(vec![(x0, y0), (x1, y1)]);
    }
    lines = clip_multi(&lines, poly).clip;

    for i in 0..lines.len() {
        lines[i] = resample(&lines[i], 2.);
    }

    binclip_multi(&lines, Rc::new(move |p, _| pattern_func(p))).clip
}

pub fn vein_shape(poly: &Polyline, n_opt: Option<i64>) -> Vec<Polyline> {
    let n = n_opt.unwrap_or(50);
    let bbox = get_boundingbox(poly);
    let mut out = vec![];
    for _ in 0..n {
        let mut x = bbox.x + rand() * bbox.w;
        let mut y = bbox.y + rand() * bbox.h;
        let mut o = vec![(x, y)];
        for _ in 0..15 {
            let dx = (noise(x * 0.1, Some(y * 0.1), Some(7.)) - 0.5) * 4.;
            let dy = (noise(x * 0.1, Some(y * 0.1), Some(6.)) - 0.5) * 4.;
            x += dx;
            y += dy;
            o.push((x, y));
        }
        out.push(o);
    }
    out = clip_multi(&out, poly).clip;
    return out;
}
pub fn smalldot_shape(poly: &Polyline, scale: f64) -> Vec<Polyline> {
    let mut samples = vec![];
    let bbox = get_boundingbox(poly);
    poissondisk(bbox.w, bbox.h, 5. * scale, &mut samples);
    for i in 0..samples.len() {
        samples[i].0 += bbox.x;
        samples[i].1 += bbox.y;
    }
    let mut out = vec![];
    let n = 7;
    for (x, y) in samples {
        let t = if (y > 0.) { (y / 300.) } else { 0.5 };
        // console.log(y,t);
        if ((t > 0.4 || y < 0.) && t > rand()) {
            continue;
        }
        for k in 0..2 {
            let mut o = vec![];
            for j in 0..n {
                let t = j / (n - 1);
                let a = t as f64 * PI * 2.;
                o.push((
                    f64::cos(a) * 1. - k as f64 * 0.3,
                    f64::sin(a) * 0.5 - k as f64 * 0.3,
                ))
            }
            out.push(trsl_poly(&rot_poly(&o, rand() * PI * 2.), x, y));
        }
    }
    clip_multi(&out, poly).clip
}

pub fn isect_circ_line((cx, cy): Point, r: f64, (x0, y0): Point, (x1, y1): Point) -> Option<f64> {
    //https://stackoverflow.com/a/1084899
    let dx = x1 - x0;
    let dy = y1 - y0;
    let fx = x0 - cx;
    let fy = y0 - cy;
    let a = dx * dx + dy * dy;
    let b = 2. * (fx * dx + fy * dy);
    let c = (fx * fx + fy * fy) - r * r;
    let mut discriminant = b * b - 4. * a * c;
    if discriminant < 0. {
        return None;
    }
    discriminant = discriminant.sqrt();
    let t0 = (-b - discriminant) / (2. * a);
    if 0. <= t0 && t0 <= 1. {
        return Some(t0);
    }
    let t = (-b + discriminant) / (2. * a);
    if t > 1. || t < 0. {
        return None;
    }
    return Some(t);
}

pub fn resample(polyline_slice: &[Point], step: f64) -> Vec<Point> {
    let mut polyline = polyline_slice.to_vec();
    if polyline_slice.len() < 2 {
        return polyline;
    }
    let mut out = vec![polyline[0]];
    let mut next;
    let mut i = 0;
    while i < polyline.len() - 1 {
        let a = polyline[i];
        let b = polyline[i + 1];
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let d = f64::sqrt(dx * dx + dy * dy);
        if d == 0. {
            i += 1;
            continue;
        }
        let n = (d / step).trunc();
        let rest = (n as f64 * step) / d;
        let rpx = a.0 * (1. - rest) + b.0 * rest;
        let rpy = a.1 * (1. - rest) + b.1 * rest;
        for j in 1..n as i64 {
            let t = j as f64 / n;
            let x = a.0 * (1. - t) + rpx * t;
            let y = a.1 * (1. - t) + rpy * t;
            // let xy = [x, y];
            // for k in 2..a.len() {
            //     xy.push(a[k] * (1 - t) + (a[k] * (1 - rest) + b[k] * rest) * t);
            // }
            out.push((x, y));
        }

        next = None;
        for j in i + 2..polyline.len() {
            let b = polyline[j - 1];
            let c = polyline[j];
            if b.0 == c.0 && b.1 == c.1 {
                continue;
            }
            let t_opt: Option<f64> = isect_circ_line((rpx, rpy), step, b, c);
            let Some(t) = t_opt else {
                continue;
            };

            let q = (b.0 * (1. - t) + c.0 * t, b.1 * (1. - t) + c.1 * t);
            // for k in 2..b.len() {
            //     q.push(b[k] * (1 - t) + c[k] * t);
            // }
            out.push(q);
            polyline[j - 1] = q;
            next = Some(j - 1);
            break;
        }
        let Some(nxt) = next else {
            break;
        };
        i = nxt;
    }

    if out.len() > 1 {
        let lx = out[out.len() - 1].0;
        let ly = out[out.len() - 1].1;
        let mx = polyline[polyline.len() - 1].0;
        let my = polyline[polyline.len() - 1].1;
        let d = f64::sqrt((mx - lx).powi(2) + (my - ly).powi(2));
        if d < step * 0.5 {
            out.pop();
        }
    }
    out.push(polyline[polyline.len() - 1]);
    return out;
}

pub fn pt_seg_dist((x, y): Point, (x1, y1): Point, (x2, y2): Point) -> f64 {
    // https://stackoverflow.com/a/6853926
    let a = x - x1;
    let b = y - y1;
    let c = x2 - x1;
    let d = y2 - y1;
    let dot = a * c + b * d;
    let len_sq = c * c + d * d;
    let mut param = -1.;
    if len_sq != 0. {
        param = dot / len_sq;
    }
    let xx;
    let yy;
    if param < 0. {
        xx = x1;
        yy = y1;
    } else if param > 1. {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * c;
        yy = y1 + param * d;
    }
    let dx = x - xx;
    let dy = y - yy;
    return f64::sqrt(dx * dx + dy * dy);
}
/*


pub fn approx_poly_dp(polyline, epsilon){
  if (polyline.len() <= 2){
    return polyline;
  }
  let dmax   = 0;
  let argmax = -1;
  for i in 1; i < polyline.len()-1 {
    let d = pt_seg_dist(polyline[i] ,
                        polyline[0] ,
                        polyline[polyline.len()-1] );
    if (d > dmax){
      dmax = d;
      argmax = i;
    }
  }
  let ret = [];
  if (dmax > epsilon){
    let L = approx_poly_dp(polyline.slice(0,argmax+1),epsilon);
    let R = approx_poly_dp(polyline.slice(argmax,polyline.len()),epsilon);
    ret = ret.concat(L.slice(0,L.len()-1)).concat(R);
  }else{
    ret.push(polyline[0].slice());
    ret.push(polyline[polyline.len()-1].slice());
  }
  return ret;
}
*/

pub fn distsq((x0, y0): Point, (x1, y1): Point) -> f64 {
    let dx = x0 - x1;
    let dy = y0 - y1;
    dx * dx + dy * dy
}

pub fn poissondisk(width: f64, height: f64, radius: f64, samples: &mut Polyline) {
    let mut active = vec![];
    let radius_over_root_2 = radius / 2.0f64.sqrt();
    let r2 = radius.powi(2);
    let cols = (width / radius_over_root_2) as usize;
    let rows = (height / radius_over_root_2) as usize;
    let mut grid: Vec<i32> = vec![-1; (cols) * (rows)];
    let mut pos = (width / 2., height / 2.);
    samples.push(pos);
    for i in 0..samples.len() {
        let col = (samples[i].0 / radius_over_root_2) as usize;
        let row = (samples[i].1 / radius_over_root_2) as usize;
        grid[col + (row * cols)] = i as i32;
        active.push(samples[i]);
    }
    while !active.is_empty() {
        let ridx = (rand() * active.len() as f64) as usize;
        pos = active[ridx];
        let mut found = false;
        for _ in 0..30 {
            let sr = radius + (rand() * radius);
            let sa = 6.2831853072 * rand();
            let sx = pos.0 + (sr * sa.cos());
            let sy = pos.1 + (sr * sa.sin());
            let col = (sx / radius_over_root_2) as i32;
            let row = (sy / radius_over_root_2) as i32;
            if col > 0
                && row > 0
                && col < cols as i32 - 1
                && row < rows as i32 - 1
                && grid[(col + (row * cols as i32)) as usize] == -1
            {
                let mut ok = true;
                for i in -1..=1 {
                    for j in -1..=1 {
                        let idx = (((row + i) * cols as i32) + col) + j;
                        let nbr = grid[idx as usize];
                        if -1 != nbr {
                            let d = distsq((sx, sy), samples[nbr as usize]);
                            if d < r2 {
                                ok = false;
                            };
                        };
                    }
                }
                if ok {
                    found = true;
                    grid[((row * (cols as i32)) + col) as usize] = samples.len() as i32;
                    let sample = (sx, sy);
                    active.push(sample);
                    samples.push(sample);
                };
            };
        }
        if !found {
            active.remove(ridx);
        };
    }
}

pub fn pow(a: f64, b: f64) -> f64 {
    return a.abs().powf(b).copysign(a);
}

pub fn gauss2d(x: f64, y: f64) -> f64 {
    let z0 = f64::exp(-0.5 * x * x);
    let z1 = f64::exp(-0.5 * y * y);
    return z0 * z1;
}

pub fn scl_poly(poly: &Polyline, sx: f64, sy_opt: Option<f64>) -> Polyline {
    let sy = sy_opt.unwrap_or(sx);
    poly.iter().map(|xy| (xy.0 * sx, xy.1 * sy)).collect()
}
pub fn shr_poly(poly: &Polyline, sx: f64) -> Polyline {
    poly.iter().map(|xy| (xy.0 + xy.1 * sx, xy.1)).collect()
}
pub fn rot_poly(poly: &Polyline, th: f64) -> Polyline {
    let costh = f64::cos(th);
    let sinth = f64::sin(th);
    poly.iter()
        .map(|(x0, y0)| (x0 * costh - y0 * sinth, x0 * sinth + y0 * costh))
        .collect()
}

pub fn pattern_dot(scale: f64) -> Rc<dyn Fn((f64, f64)) -> bool> {
    let mut samples = vec![];
    poissondisk(500., 300., 20. * scale, &mut samples);
    let mut rs = vec![];
    for _ in 0..samples.len() {
        rs.push((rand() * 5. + 10.) * scale)
    }
    Rc::new(move |(x, y)| {
        for i in 0..samples.len() {
            let r = rs[i];
            if dist((x, y), samples[i]) < r {
                let (x0, y0) = samples[i];
                let dx = x - x0;
                let dy = y - y0;
                if gauss2d(dx / r * 2., dy / r * 2.) * noise(x, Some(y), Some(999.)) > 0.2 {
                    return true;
                }
            }
        }
        return false;
    })
}
