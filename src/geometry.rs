use crate::custom_rand::rand;
use std::{
    collections::HashMap,
    f64::consts::{E, PI},
    iter::successors,
};

pub type Point = (f64, f64);
pub type Polyline = Vec<Point>;

pub fn dist((x0, y0): Point, (x1, y1): Point) -> f64 {
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a * (1. - t) + b * t
}
pub fn lerp2d((x0, y0): Point, (x1, y1): Point, t: f64) -> Point {
    (x0 * (1. - t) + x1 * t, y0 * (1. - t) + y1 * t)
}

struct BoundingBox {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
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
    side: i32,
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

pub fn get_side((x, y): Point, (x0, y0): Point, (x1, y1): Point) -> i32 {
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
        isect_mir: &mut HashMap<(usize, usize, i32), (usize, usize, i32)>,
    ) {
        let n = verts0.len();
        for i in 0..n {
            let m = verts0[i].isects.len();
            for j in 0..m {
                let id = (idx, i, j as i32);
                let jump = verts0[i].isects[j].jump.unwrap_or(false);
                let jd = if jump { 1 - idx } else { idx };
                let k = verts0[i].isects[j].other.unwrap();
                let z = (if jump { &verts1 } else { &verts0 })[k]
                    .isects
                    .iter()
                    .position(|x| (x.jump == Some(jump) && x.other == Some(i)))
                    .unwrap();
                isect_mir.insert(id, (jd, k, z as i32));
            }
        }
    }
    mirror_isects(&mut verts0, &mut verts1, 0, &mut isect_mir);
    mirror_isects(&mut verts1, &mut verts0, 1, &mut isect_mir);

    pub fn trace_outline(
        idx: usize,
        i0: usize,
        j0: i32,
        dir: i32,
        verts0: &Vec<Vertex>,
        verts1: &Vec<Vertex>,
        isect_mir: &HashMap<(usize, usize, i32), (usize, usize, i32)>,
    ) -> Option<Polyline> {
        pub fn trace_from(
            mut zero: Option<(usize, usize, i32)>,
            verts0: &Vec<Vertex>,
            verts1: &Vec<Vertex>,
            idx: usize,
            i0: usize,
            j0: i32,
            dir: i32,
            out: &mut Polyline,
            isect_mir: &HashMap<(usize, usize, i32), (usize, usize, i32)>,
        ) -> bool {
            if zero == None {
                zero = Some((idx, i0, j0));
            } else if idx == zero.unwrap().0 && i0 == zero.unwrap().1 && j0 == zero.unwrap().2 {
                return true;
            }
            let verts = if idx > 0 { verts1 } else { verts0 };
            let n = verts.len();
            let p = &verts[i0];
            let i1 = (((i0 + n) as i32) + dir) as usize % n;
            if j0 == -1 {
                out.push(p.xy);
                if dir < 0 {
                    return trace_from(
                        zero,
                        verts0,
                        verts1,
                        idx,
                        i1,
                        verts[i1].isects.len() as i32 - 1,
                        dir,
                        out,
                        isect_mir,
                    );
                } else if verts[i0].isects.is_empty() {
                    return trace_from(zero, verts0, verts1, idx, i1, -1, dir, out, isect_mir);
                } else {
                    return trace_from(zero, verts0, verts1, idx, i0, 0, dir, out, isect_mir);
                }
            } else if j0 >= p.isects.len() as i32 {
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
                        (z - 1) as i32,
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
                    (z + 1) as i32,
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

    pub fn check_concavity(poly: &Polyline, idx: usize) -> i32 {
        let n = poly.len();
        let a = poly[(idx - 1 + n) % n];
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
    clip: Vec<Polyline>,
    dont_clip: Vec<Polyline>,
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
    if (polyline.is_empty()) {
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

pub fn clip_multi(
    polylines: &Vec<Polyline>,
    polygon: &Polyline,
    clipper_func_opt: Option<fn(&Polyline, &Polyline) -> ClipSegments>,
) -> ClipSegments {
    let clipper_func = clipper_func_opt.unwrap_or(clip);
    let mut out = ClipSegments::default();
    for polyline in polylines {
        out.extend(clipper_func(polyline, polygon));
    }
    return out;
}

pub fn binclip(polyline: &Polyline, func: fn(Point, usize) -> bool) -> ClipSegments {
    if (polyline.is_empty()) {
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

    lines = clip_multi(&lines, poly, None).clip;

    let carve = trsl_poly(poly, -dx, -dy);

    lines = clip_multi(&lines, &carve, None).dont_clip;

    for i in 0..lines.len() {
        let line = &lines[i];
        let mut a = line[0];
        let mut b = line[1];
        let s = (rand() as f64) * 0.5;
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
