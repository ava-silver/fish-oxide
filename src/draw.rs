use std::f64::consts::PI;

use crate::{
    custom_rand::{deviate, noise},
    geometry::{
        clip_multi, dist, get_boundingbox, lerp, lerp2d, poly_union, resample, trsl_poly, Polyline,
    },
    hershey::compile_hershey,
    params::Params,
};

/*
pub pub fn draw_svg(polylines){
  let o = `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320">`
  o += `<rect x="0" y="0" width="520" height="320" fill="floralwhite"/><rect x="10" y="10" width="500" height="300" stroke="black" stroke-width="1" fill="none"/><path stroke="black" stroke-width="1" fill="none" stroke-linecap="round" stroke-linejoin="round" d="`
  for (let i = 0; i < polylines.length; i++){
    o += '\nM ';
    for (let j = 0; j < polylines[i].length; j++){
      let [x,y] = polylines[i][j];
      o += `${(~~((x+10)*100)) /100} ${(~~((y+10)*100)) /100} `;
    }
  }
  o += `\n"/></svg>`
  return o;
}

pub pub fn draw_svg_anim(polylines,speed){
  let o = `<svg xmlns="http://www.w3.org/2000/svg" width="520" height="320">`;
  o += `<rect x="0" y="0" width="520" height="320" fill="floralwhite"/><rect x="10" y="10" width="500" height="300" stroke="black" stroke-width="1" fill="none"/>`
  let lengths = [];
  let acc_lengths = [];
  let total_l = 0;
  for (let i = 0; i < polylines.length; i++){
    let l = 0;
    for (let j = 1; j < polylines[i].length; j++){
      l += Math.hypot(
        polylines[i][j-1][0]-polylines[i][j][0],
        polylines[i][j-1][1]-polylines[i][j][1]
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


pub fn fish_body_a(curve0,curve1,scale_scale,pattern_func){
  let curve2 = [];
  let curve3 = [];
  for i in 0..curve0.len() {
    curve2.push(lerp2d(...curve0[i],...curve1[i],0.95));
    curve3.push(lerp2d(...curve0[i],...curve1[i],0.85));
  }
  let outline1 = curve0.concat(curve1.slice().reverse());
  let outline2 = curve0.concat(curve2.slice().reverse());
  let outline3 = curve0.concat(curve3.slice().reverse());

  let bbox = get_boundingbox(curve0.concat(curve1));
  let m = !!(bbox.w/(scale_scale*15));
  let n = !!(bbox.h/(scale_scale*15));
  let uw = bbox.w/m;
  let uh = bbox.h/n;

  let pub fn = pattern_func?((x,y,w,h)=>squama(w,h,Number(pattern_func(x,y))*3)):((x,y,w,h)=>squama(w,h))
  let sq = squama_mesh(m,n+3,uw,uh,pub fn,uw*3,uh*3,true).map(a=>trsl_poly(a,bbox.x,bbox.y-uh*1.5));
  let o0 = clip_multi(sq,outline2)[true];
  let o1 = clip_multi(o0,outline3);
  o1.dont_clip = o1.dont_clip.filter(x=>rand()<0.6);
  let o = [];
  o.push(curve0,curve1.slice().reverse(),...o1.clip,...o1.dont_clip);
  return o;
}

pub fn fish_body_b(curve0,curve1,scale_scale,pattern_func){
  let curve2 = [];
  for i in 0..curve0.len() {
    curve2.push(lerp2d(...curve0[i],...curve1[i],0.95));
  }
  let outline1 = curve0.concat(curve1.slice().reverse());
  let outline2 = curve0.concat(curve2.slice().reverse());

  let bbox = get_boundingbox(curve0.concat(curve1));
  let m = !!(bbox.w/(scale_scale*5));
  let n = !!(bbox.h/(scale_scale*5));
  let uw = bbox.w/m;
  let uh = bbox.h/n;

  let sq = squama_mesh(m,n+16,uw,uh,(x,y,w,h)=>squama(w*0.7,h*0.6,0),uw*8,uh*8,false).map(a=>trsl_poly(a,bbox.x,bbox.y-uh*8));
  let o0 = clip_multi(sq,outline2)[true];

  let o1 = [];
  for i in 0..o0.len() {
    let [x,y] = o0[i][0];
    let t = (y-bbox.y)/bbox.h;
    // if (rand() > t){
    //   o1.push(o0[i]);
    // }
    // if ((!!(x/30))%2 || (rand() > t && rand()>t)){
    //   o1.push(o0[i]);
    // }
    if (pattern_func){
      if (pattern_func(x,y) || (rand() > t && rand()>t)) {
        o1.push(o0[i]);
      }
    }else{
      if (rand() > t){
        o1.push(o0[i]);
      }
    }
  }
  let o = [];
  o.push(curve0,curve1.slice().reverse(),...o1);
  return o;
}


pub fn ogee(x){
  return 4 * f64::powf(x-0.5,3) + 0.5;
}

pub fn fish_body_c(curve0,curve1,scale_scale){
  let step = 6*scale_scale;

  let curve2 = [];
  let curve3 = [];

  for i in 0..curve0.len() {
    curve2.push(lerp2d(...curve0[i],...curve1[i],0.95));
    curve3.push(lerp2d(...curve0[i],...curve1[i],0.4));
  }
  let outline1 = curve0.concat(curve1.slice().reverse());
  let outline2 = curve0.concat(curve2.slice().reverse());

  let bbox = get_boundingbox(curve0.concat(curve1));
  bbox.x -= step;
  bbox.y -= step;
  bbox.w += step*2;
  bbox.h += step*2;

  let lines = [curve3.reverse()];

  for i in -bbox.h; i < bbox.w; i+=step){
    let x0 = bbox.x + i;
    let y0 = bbox.y;
    let x1 = bbox.x + i + bbox.h;
    let y1 = bbox.y + bbox.h;
    lines.push([[x0,y0],[x1,y1]]);
  }
  for i in 0..bbox.w+bbox.h; i+=step){
    let x0 = bbox.x + i;
    let y0 = bbox.y;
    let x1 = bbox.x + i - bbox.h;
    let y1 = bbox.y + bbox.h;
    lines.push([[x0,y0],[x1,y1]]);
  }
  for i in 0..lines.len() {
    lines[i] = resample(lines[i],4);
    for j in 0;j < lines[i].len(); j++){
      let [x,y] = lines[i][j];
      let t = (y-bbox.y)/bbox.h;
      let y1 = -Math.cos(t*PI)*bbox.h/2+bbox.y+bbox.h/2;

      let dx = (noise(x*0.005,y1*0.005,0.1)-0.5)*50;
      let dy = (noise(x*0.005,y1*0.005,1.2)-0.5)*50;

      lines[i][j][0] += dx;
      lines[i][j][1] = y1 + dy;
    }
  }

  let o0 = clip_multi(lines,outline2)[true];

  o0 = clip_multi(o0,(x,y,t)=>(rand()>t||rand()>t),binclip).clip;


  let o = [];

  o.push(curve0,curve1.slice().reverse(),...o0);
  return o;
}

pub fn fish_body_d(curve0,curve1,scale_scale){
  let curve2 = [];
  for i in 0..curve0.len() {
    curve2.push(lerp2d(...curve0[i],...curve1[i],0.4));
  }
  curve0 = resample(curve0,10*scale_scale);
  curve1 = resample(curve1,10*scale_scale);
  curve2 = resample(curve2,10*scale_scale);

  let outline1 = curve0.concat(curve1.slice().reverse());
  let outline2 = curve0.concat(curve2.slice().reverse());

  let o0 = [curve2];
  for i in 3; i < Math.min(curve0.len(),curve1.len(),curve2.len()) {
    let a = [curve0[i],curve2[i-3]];
    let b = [curve2[i-3],curve1[i]];

    o0.push(a,b);
  }

  let o1 = [];
  for i in 0..o0.len() {
    o0[i] = resample(o0[i],4);
    for j in 0..o0[i].len(); j++){
      let [x,y] = o0[i][j];
      let dx = 30*(noise(x*0.01,y*0.01,-1)-0.5);
      let dy = 30*(noise(x*0.01,y*0.01,9)-0.5);
      o0[i][j][0] += dx;
      o0[i][j][1] += dy;
    }

    o1.push(...binclip(o0[i],(x,y,t)=>(
      (rand()>Math.cos(t*PI) && rand() < x/500) || (rand()>Math.cos(t*PI) && rand() < x/500)
    )).clip);
  }
  o1 = clip_multi(o1,outline1).clip;

  let sh = vein_shape(outline1);

  let o = [];
  o.push(curve0,curve1.slice().reverse(),...o1,...sh);
  return o;
}




pub fn fin_a(curve,ang0,ang1,func,clip_root=false,curvature0=0,curvature1=0,softness=10){
  let angs = [];
  for i in 0..curve.len() {

    if (i == 0){
      angs.push( Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]) - PI/2 );
    }else if (i == curve.len()-1){
      angs.push( Math.atan2(curve[i][1]-curve[i-1][1], curve[i][0]-curve[i-1][0]) - PI/2 );
    }else{
      let a0 = Math.atan2(curve[i-1][1]-curve[i][1], curve[i-1][0]-curve[i][0]);
      let a1 = Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]);
      while (a1 > a0){
        a1 -= PI*2;
      }
      a1 += PI*2;
      let a = (a0+a1)/2;
      angs.push(a);
    }
  }
  let out0 = [];
  let out1 = [];
  let out2 = [];
  let out3 = [];
  for i in 0..curve.len() {
    let t = i/(curve.len()-1);
    let aa = lerp(ang0,ang1,t);
    let a = angs[i]+aa;
    let w = func(t);

    let [x0,y0] = curve[i];
    let x1 = x0 + Math.cos(a)*w;
    let y1 = y0 + f64::sin(a)*w;

    let p = resample([[x0,y0],[x1,y1]],3);
    for j in 0..p.len(); j++){
      let s = j/(p.len()-1);
      let ss = Math.sqrt(s);
      let [x,y] = p[j];
      let cv = lerp(curvature0,curvature1,t)*f64::sin(s*PI);
      p[j][0] += noise(x*0.1,y*0.1,3)*ss*softness + Math.cos(a-PI/2)*cv;
      p[j][1] += noise(x*0.1,y*0.1,4)*ss*softness + f64::sin(a-PI/2)*cv;
    }
    if (i == 0){
      out2 = p;
    }else if (i == curve.len()-1){
      out3 = p.slice().reverse();
    }else{
      out0.push(p[p.len()-1]);
      // if (i % 2){
        let q = p.slice(clip_root?(   !!(rand()*4)  ):0,Math.max(2,!!(p.len()*(rand()*0.5+0.5))));
        if (q.len()){
          out1.push(q);
        }
      // }
    }
  }
  out0 = resample(out0,3);
  for i in 0..out0.len() {
    let [x,y] = out0[i];
    out0[i][0] += (noise(x*0.1,y*0.1)*6-3)*(softness/10);
    out0[i][1] += (noise(x*0.1,y*0.1)*6-3)*(softness/10);
  }
  let o = out2.concat(out0).concat(out3);
  out1.unshift(o);
  return [o.concat(curve.slice().reverse()),out1];
}


pub fn fin_b(curve,ang0,ang1,func,dark=1){
  let angs = [];
  for i in 0..curve.len() {

    if (i == 0){
      angs.push( Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]) - PI/2 );
    }else if (i == curve.len()-1){
      angs.push( Math.atan2(curve[i][1]-curve[i-1][1], curve[i][0]-curve[i-1][0]) - PI/2 );
    }else{
      let a0 = Math.atan2(curve[i-1][1]-curve[i][1], curve[i-1][0]-curve[i][0]);
      let a1 = Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]);
      while (a1 > a0){
        a1 -= PI*2;
      }
      a1 += PI*2;
      let a = (a0+a1)/2;
      angs.push(a);
    }
  }

  let out0 = [];
  let out1 = [];
  let out2 = [];
  let out3 = [];
  for i in 0..curve.len() {
    let t = i/(curve.len()-1);
    let aa = lerp(ang0,ang1,t);
    let a = angs[i]+aa;
    let w = func(t);

    let [x0,y0] = curve[i];
    let x1 = x0 + Math.cos(a)*w;
    let y1 = y0 + f64::sin(a)*w;

    let b = [
      x1 + 0.5 * Math.cos(a-PI/2),
      y1 + 0.5 * f64::sin(a-PI/2),
    ];
    let c = [
      x1 + 0.5 * Math.cos(a+PI/2),
      y1 + 0.5 * f64::sin(a+PI/2),
    ];

    let p = [
      curve[i][0] + 1.8 * Math.cos(a-PI/2),
      curve[i][1] + 1.8 * f64::sin(a-PI/2),
    ];
    let q = [
      curve[i][0] + 1.8 * Math.cos(a+PI/2),
      curve[i][1] + 1.8 * f64::sin(a+PI/2),
    ];
    out1.push([x1,y1]);
    out0.push([p,b,c,q]);
  }

  let n = 10;
  for i in 0..curve.len()-1 {

    let [_,__,a0,q0] = out0[i];
    let [p1,a1,___,____] = out0[i+1];

    let b = lerp2d(...a0,...q0,0.1);
    let c = lerp2d(...a1,...p1,0.1);

    let o = [];
    let ang = Math.atan2(c[1]-b[1],c[0]-b[0]);

    for j in 0..n; j++){
      let t = j/(n-1);
      let d = f64::sin(t*PI)*2;
      let a = lerp2d(...b,...c,t);
      o.push([
        a[0] + Math.cos(ang+PI/2)*d,
        a[1] + f64::sin(ang+PI/2)*d,
      ])
    }

    // out2.push([b,c]);
    out2.push(o);

    let m = !!( Math.min(dist(...a0,...q0),dist(...a1,...p1) ) /10 * dark);
    let e = lerp2d(...curve[i],...curve[i+1],0.5);
    for k in 0; k < m; k ++){
      let p = [];
      let s= k/m*0.7;
      for j in 1; j < n-1; j++){
        p.push(lerp2d(...o[j],...e,s));
      }
      out3.push(p);
    }
  }

  let out4 = [];
  if (out0.len() > 1){
    let clipper = out0[0];
    out4.push(out0[0])
    for i in 1; i < out0.len() {
      out4.push(...clip(out0[i],clipper).dont_clip);
      clipper = poly_union(clipper,out0[i]);
    }
  }

  return [out2.flat().concat(curve.slice().reverse()),out4.concat(out2).concat(out3)];
}


pub fn finlet(curve,h,dir=1){
  let angs = [];
  for i in 0..curve.len() {
    if (i == 0){
      angs.push( Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]) - PI/2 );
    }else if (i == curve.len()-1){
      angs.push( Math.atan2(curve[i][1]-curve[i-1][1], curve[i][0]-curve[i-1][0]) - PI/2 );
    }else{
      let a0 = Math.atan2(curve[i-1][1]-curve[i][1], curve[i-1][0]-curve[i][0]);
      let a1 = Math.atan2(curve[i+1][1]-curve[i][1], curve[i+1][0]-curve[i][0]);
      while (a1 > a0){
        a1 -= PI*2;
      }
      a1 += PI*2;
      let a = (a0+a1)/2;
      angs.push(a);
    }
  }
  let out0 = [];
  for i in 0..curve.len() {
    let t = i/(curve.len()-1);
    let a = angs[i];
    let w = (i+1) % 3 ? 0 : h;
    if (dir > 0){
      w *= (1-t*0.5);
    }else{
      w *= 0.5 + t * 0.5
    }

    let [x0,y0] = curve[i];
    let x1 = x0 + Math.cos(a)*w;
    let y1 = y0 + f64::sin(a)*w;
    out0.push([x1,y1]);

  }
  out0 = resample(out0,2);
  for i in 0..out0.len() {
    let [x,y] = out0[i];
    out0[i][0] += noise(x*0.1,y*0.1)*2-3;
    out0[i][1] += noise(x*0.1,y*0.1)*2-3;
  }
  out0.push(curve[curve.len()-1]);
  return [out0.concat(curve.slice().reverse()),[out0]];
}

pub fn fin_adipose(curve,dx,dy,r){
  let n = 20;
  let [x0,y0] = curve[!!(curve.len()/2)];
  let [x,y] = [x0+dx,y0+dy];
  let [x1,y1] = curve[0];
  let [x2,y2] = curve[curve.len()-1];
  let d1 = dist(x,y,x1,y1);
  let d2 = dist(x,y,x2,y2);
  let a1 = Math.acos(r/d1);
  let a2 = Math.acos(r/d2);
  let a01 = Math.atan2(y1-y,x1-x)+a1;
  let a02 = Math.atan2(y2-y,x2-x)-a2;
  a02 -= PI*2;
  while (a02 < a01){
    a02 += PI*2;
  }
  let out0 = [[x1,y1]]
  for i in 0..n {
    let t = i/(n-1);
    let a = lerp(a01,a02,t);
    let p = [
      x+Math.cos(a)*r,
      y+f64::sin(a)*r,
    ]
    out0.push(p);
  }
  out0.push([x2,y2]);
  out0 = resample(out0,3);
  for i in 0..out0.len() {
    let t = i/(out0.len()-1);
    let s = f64::sin(t*PI);
    let [x,y] = out0[i];
    out0[i][0] += (noise(x*0.01,y*0.01)-0.5)*s*50;
    out0[i][1] += (noise(x*0.01,y*0.01)-0.5)*s*50;
  }
  let cc = out0.concat(curve.slice().reverse());
  let out1 = clip(trsl_poly(out0,0,4),cc).clip;

  out1 = clip_multi(out1,(x,y,t)=>(rand()<f64::sin(t*PI)),binclip).clip;
  return [cc,[out0,...out1]];

}


pub fn fish_lip(x0,y0,x1,y1,w){
  x0 += rand()*0.001-0.0005;
  y0 += rand()*0.001-0.0005;
  x1 += rand()*0.001-0.0005;
  y1 += rand()*0.001-0.0005;
  let h = dist(x0,y0,x1,y1);
  let a0 = Math.atan2(y1-y0,x1-x0);
  let n = 10;
  let ang = Math.acos(w/h);
  let dx = Math.cos(a0+PI/2)*0.5;
  let dy = f64::sin(a0+PI/2)*0.5;
  let o = [[x0-dx,y0-dy]];
  for i in 0..n {
    let t = i/(n-1);
    let a = lerp(ang,PI*2-ang,t) + a0;
    let x = -Math.cos(a)*w + x1;
    let y = -f64::sin(a)*w + y1;
    o.push([x,y]);
  }
  o.push([x0+dx,y0+dy]);
  o = resample(o,2.5);
  for i in 0..o.len() {
    let [x,y] = o[i];
    o[i][0] += noise(x*0.05,y*0.05,-1)*2-1;
    o[i][1] += noise(x*0.05,y*0.05,-2)*2-1;
  }
  return o;
}

pub fn fish_teeth(x0,y0,x1,y1,h,dir,sep=3.5){
  let n = Math.max(2,!!(dist(x0,y0,x1,y1)/sep));
  let ang = Math.atan2(y1-y0,x1-x0);
  let out = [];
  for i in 0..n {
    let t = i/(n-1);
    let a = lerp2d(x0,y0,x1,y1,t);
    let w = h*t;
    let b = [
      a[0]+Math.cos(ang+dir*PI/2)*w,
      a[1]+f64::sin(ang+dir*PI/2)*w,
    ];
    let c = [
      a[0] + 1 * Math.cos(ang),
      a[1] + 1 * f64::sin(ang),
    ];
    let d = [
      a[0] + 1 * Math.cos(ang+PI),
      a[1] + 1 * f64::sin(ang+PI),
    ];
    let e = lerp2d(...c,...b,0.7);
    let f = lerp2d(...d,...b,0.7);
    let g = [
      a[0]+Math.cos(ang+dir*(PI/2+0.15))*w,
      a[1]+f64::sin(ang+dir*(PI/2+0.15))*w,
    ]
    out.push([c,e,g,f,d])
    // out.push(barbel(...a,10,ang+dir*PI/2))
  }
  return out;
}

pub fn fish_jaw(x0,y0,x1,y1,x2,y2){
  let n = 10;
  let ang = Math.atan2(y2-y0,x2-x0);
  let d = dist(x0,y0,x2,y2);
  let o = [];
  for i in 0..n {
    let t = i/(n-1);
    let s = f64::sin(t*PI);
    let w = s*d/20;
    let p = lerp2d(x2,y2,x0,y0,t);
    let q = [
      p[0] + Math.cos(ang-PI/2)*w,
      p[1] + f64::sin(ang-PI/2)*w,
    ]
    let qq = [
      q[0] + (noise(q[0]*0.01,q[1]*0.01,1)-0.5)*4*s,
      q[1] + (noise(q[0]*0.01,q[1]*0.01,4)-0.5)*4*s,
    ];
    o.push(qq);
  }
  return [[[x2,y2],[x1,y1],[x0,y0]],[o,...vein_shape(o,5)]];
}


pub fn fish_eye_a(ex,ey,rad){
  let n = 20;
  let eye0 = [];
  let eye1 = [];
  let eye2 = [];
  for i in 0..n {
    let t = i/(n-1);
    let a = t * PI*2 + Math.PI/4*3;
    eye0.push([
      ex + Math.cos(a)*rad,
      ey + f64::sin(a)*rad
    ]);
    if (t > 0.5){
      eye1.push([
        ex + Math.cos(a)*(rad*0.8),
        ey + f64::sin(a)*(rad*0.8)
      ]);
    }
    eye2.push([
      ex + Math.cos(a)*(rad*0.4) -0.75,
      ey + f64::sin(a)*(rad*0.4) -0.75
    ]);
  }

  let ef = shade_shape(eye2,2.7,10,10);
  return [eye0,[eye0,eye1,eye2,...ef]];
}

pub fn fish_eye_b(ex,ey,rad){
  let n = 20;
  let eye0 = [];
  let eye1 = [];
  let eye2 = [];
  for i in 0..n {
    let t = i/(n-1);
    let a = t * PI*2+E;
    eye0.push([
      ex + Math.cos(a)*rad,
      ey + f64::sin(a)*rad
    ]);
    eye2.push([
      ex + Math.cos(a)*(rad*0.4),
      ey + f64::sin(a)*(rad*0.4)
    ]);
  }
  let m =!!((rad*0.6)/2);
  for i in 0..m {
    let r = rad - i * 2;
    let e = [];
    for i in 0..n {
      let t = i/(n-1);
      let a = lerp(PI*7/8,PI*13/8,t)
      e.push([
        ex + Math.cos(a)*r,
        ey + f64::sin(a)*r
      ]);

    }
    eye1.push(e);
  }
  let trig = [
    [ex+Math.cos(-PI*3/4)*(rad*0.9),ey+f64::sin(-PI*3/4)*(rad*0.9)],
    [ex+1,ey+1],
    [ex+Math.cos(-PI*11/12)*(rad*0.9),ey+f64::sin(-PI*11/12)*(rad*0.9)],
  ];
  trig = resample(trig,3);
  for i in 0..trig.len() {
    let [x,y] = trig[i];
    x += noise(x*0.1,y*0.1,22)*4-2;
    y += noise(x*0.1,y*0.1,33)*4-2;
    trig[i] = [x,y];
  }


  let ef = fill_shape(eye2,1.5);

  ef = clip_multi(ef,trig).dont_clip;
  eye1 = clip_multi(eye1,trig).dont_clip;
  eye2 = clip(eye2,trig).dont_clip;

  return [eye0,[eye0,...eye1,...eye2,...ef]];
}



pub fn barbel(x,y,n,ang,dd=3){
  let curve = [[x,y]];
  let sd = rand()*PI*2;
  let ar = 1;
  for i in 0..n {
    x += Math.cos(ang)*dd;
    y += f64::sin(ang)*dd;
    ang += (noise(i*0.1,sd)-0.5)*ar;
    if (i < n/2){
      ar *= 1.02;
    }else{
      ar *= 0.92;
    }
    curve.push([x,y]);
  }
  let o0 = [];
  let o1 = [];
  for i in 0..n-1 {
    let t = i/(n-1);
    let w = 1.5*(1-t);

    let a = curve[i-1];
    let b = curve[i];
    let c = curve[i+1];

    let a1 = Math.atan2(c[1]-b[1],c[0]-b[0]);
    let a2;

    if (a){
      let a0 = Math.atan2(a[1]-b[1],a[0]-b[0]);

      a1 -= PI*2;
      while (a1 < a0){
        a1 += PI*2;
      }
      a2 = (a0+a1)/2;
    }else{
      a2 = a1-PI/2;
    }

    o0.push([
      b[0]+Math.cos(a2)*w,
      b[1]+f64::sin(a2)*w
    ])
    o1.push([
      b[0]+Math.cos(a2+PI)*w,
      b[1]+f64::sin(a2+PI)*w
    ])
  }
  o0.push(curve[curve.len()-1]);
  return o0.concat(o1.slice().reverse());
}

pub fn fish_head(x0,y0,x1,y1,x2,y2,arg){
  let n = 20;
  let curve0 = [];
  let curve1 = [];
  let curve2 = [];
  for i in 0..n {
    let t = i/(n-1);
    let a = PI/2 * t;
    let x = x1-pow(Math.cos(a),1.5)*(x1-x0);
    let y = y0-pow(f64::sin(a),1.5)*(y0-y1);
    // let x = lerp(x0,x1,t);
    // let y = lerp(y0,y1,t);

    let dx = (noise(x*0.01,y*0.01,9)*40-20)*(1.01-t);
    let dy = (noise(x*0.01,y*0.01,8)*40-20)*(1.01-t);
    curve0.push([x+dx,y+dy]);
  }
  for i in 0..n {
    let t = i/(n-1);
    let a = PI/2 * t;
    let x = x2-pow(Math.cos(a),0.8)*(x2-x0);
    let y = y0+pow(f64::sin(a),1.5)*(y2-y0);

    let dx = (noise(x*0.01,y*0.01,9)*40-20)*(1.01-t);
    let dy = (noise(x*0.01,y*0.01,8)*40-20)*(1.01-t);
    curve1.unshift([x+dx,y+dy]);
  }
  let ang = Math.atan2(y2-y1,x2-x1);
  for i in 1; i < n-1 {
    let t = i/(n-1);
    let p = lerp2d(x1,y1,x2,y2,t);
    let s = pow(f64::sin(t*Math.PI),0.5);
    let r = noise(t*2,1.2) * s * 20;

    let dx = Math.cos(ang-Math.PI/2) * r;
    let dy = f64::sin(ang-Math.PI/2) * r;
    curve2.push([ p[0]+dx,p[1]+dy ])
  }
  let outline = curve0.concat(curve2).concat(curve1);

  let inline = curve2.slice(!!(curve2.len()/3)).concat(curve1.slice(0,!!(curve1.len()/2))).slice(0,curve0.len());
  for i in 0..inline.len() {
    let t = i/(inline.len()-1);
    let s = f64::sin(t*PI)**2*0.1+0.12;
    inline[i] = lerp2d(...inline[i],...curve0[i],s);
  }
  let dix = (x0-inline[inline.len()-1][0])*0.3;
  let diy = (y0-inline[inline.len()-1][1])*0.2;
  for i in 0..inline.len() {
    inline[i][0] += dix;
    inline[i][1] += diy;
  }


  let par = [0.475,0.375];
  let ex = x0*par[0] + x1*par[1] + x2*(1-par[0]-par[1]);
  let ey = y0*par[0] + y1*par[1] + y2*(1-par[0]-par[1]);
  let d0 = pt_seg_dist([ex,ey],[x0,y0],[x1,y1]);
  let d1 = pt_seg_dist([ex,ey],[x0,y0],[x2,y2]);
  if (d0 < arg.eye_size && d1 < arg.eye_size){
    arg.eye_size = Math.min(d0,d1);
  }else if (d0 < arg.eye_size){
    let ang = Math.atan2(y1-y0,x1-x0)+PI/2;
    ex = x0*0.5+x1*0.5 + Math.cos(ang)*arg.eye_size;
    ey = y0*0.5+y1*0.5 + f64::sin(ang)*arg.eye_size;
  }

  let jaw_pt0 = curve1[18-arg.mouth_size];
  let jaw_l = dist(...jaw_pt0,...curve1[18])*arg.jaw_size;
  let jaw_ang0 = Math.atan2(curve1[18][1]-jaw_pt0[1],curve1[18][0]-jaw_pt0[0]);
  let jaw_ang =jaw_ang0- (arg.has_teeth*0.5+0.5)*arg.jaw_open*PI/4;
  let jaw_pt1 = [
    jaw_pt0[0]+Math.cos(jaw_ang)*jaw_l,
    jaw_pt0[1]+f64::sin(jaw_ang)*jaw_l,
  ]

  let [eye0,ef] = (arg.eye_type?fish_eye_b:fish_eye_a)(ex,ey,arg.eye_size);
  ef = clip_multi(ef,outline).clip;

  let inlines = clip(inline,eye0).dont_clip;

  let lip0 = fish_lip(...jaw_pt0,...curve1[18],3);

  let lip1 = fish_lip(...jaw_pt0,...jaw_pt1,3);

  let [jc,jaw] = fish_jaw(...curve1[15-arg.mouth_size],...jaw_pt0,...jaw_pt1);

  jaw = clip_multi(jaw,lip1).dont_clip;
  jaw = clip_multi(jaw,outline).dont_clip;

  let teeth0s = [];
  let teeth1s = [];
  if (arg.has_teeth){
    let teeth0 = fish_teeth(...jaw_pt0,...curve1[18],arg.teeth_length,-1,arg.teeth_space);
    let teeth1 = fish_teeth(...jaw_pt0,...jaw_pt1,    arg.teeth_length,1,arg.teeth_space);

    teeth0s = clip_multi(teeth0,lip0).dont_clip;
    teeth1s = clip_multi(teeth1,lip1).dont_clip;
  }

  let olines = clip(outline,lip0).dont_clip;

  let lip0s = clip(lip0,lip1).dont_clip;

  let sh = shade_shape(outline,6,-6,-6);
  sh = clip_multi(sh,lip0).dont_clip;
  sh = clip_multi(sh,eye0).dont_clip;

  let sh2 = vein_shape(outline,arg.head_texture_amount);

  // let sh2 = patternshade_shape(outline,3,(x,y)=>{
  //   return noise(x*0.1,y*0.1)>0.6;
  // })

  sh2 = clip_multi(sh2,lip0).dont_clip;
  sh2 = clip_multi(sh2,eye0).dont_clip;

  let bbs = [];

  lip1s = [lip1];

  if (arg.has_moustache){
    let bb0 = barbel(...jaw_pt0,arg.moustache_length,PI*3/4,1.5);
    lip1s = clip(lip1,bb0).dont_clip;
    jaw = clip_multi(jaw,bb0).dont_clip;
    bbs.push(bb0);
  }

  if (arg.has_beard){
    let jaw_pt;
    if (jaw[0] && jaw[0].len()){
      jaw_pt = jaw[0][!!(jaw[0].len()/2)];
    }else{
      jaw_pt = curve1[8];
    }
    let bb1 = trsl_poly(barbel(...jaw_pt,arg.beard_length,PI*0.6+rand()*0.4-0.2),rand()*1-0.5,rand()*1-0.5);
    let bb2 = trsl_poly(barbel(...jaw_pt,arg.beard_length,PI*0.6+rand()*0.4-0.2),rand()*1-0.5,rand()*1-0.5);
    let bb3 = trsl_poly(barbel(...jaw_pt,arg.beard_length,PI*0.6+rand()*0.4-0.2),rand()*1-0.5,rand()*1-0.5);

    let bb3c = clip_multi([bb3],bb2).dont_clip;
    bb3c = clip_multi(bb3c,bb1).dont_clip;
    let bb2c = clip_multi([bb2],bb1).dont_clip;
    bbs.push(bb1,...bb2c,...bb3c);
  }

  let outlinel = [[0,0],[curve0[curve0.len()-1][0],0],curve0[curve0.len()-1],...curve2,curve1[0],[curve1[0][0],300],[0,300]];


  return [outlinel,[
    ...olines,...inlines,
    ...lip0s,...lip1s,
    ...ef,...sh,...sh2,
    ...bbs,
    ...teeth0s,...teeth1s,
    ...jaw]];
}

*/
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
    if (arg.body_curve_type == 0) {
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
    } else if (arg.body_curve_type == 1) {
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
    let mut outline = curve0.clone();
    outline.extend(curve1.clone().into_iter().rev());
    let sh = todo!("shade_shape(outline,8,-12,-12)");

    let mut pattern_func: Option<Box<dyn Fn(f64, f64) -> bool>> = None;
    if (arg.pattern_type == 0) {
        //none
    } else if (arg.pattern_type == 1) {
        // pattern_func = (x,y)=>{
        //   return noise(x*0.1,y*0.1)>0.55;
        // };
        pattern_func = Some(todo!("pattern_dot(arg.pattern_scale)"));
    } else if (arg.pattern_type == 2) {
        pattern_func = Some(Box::new(|x, y| {
            (noise(x * 0.1, Some(y * 0.1), None) * f64::max(0.35, (y - 10.) / 280.)) < 0.2
        }));
    } else if (arg.pattern_type == 3) {
        pattern_func = Some(Box::new(|x, y| {
            let dx = noise(x * 0.01, Some(y * 0.01), None) * 30.;
            ((x + dx) / (30. * arg.pattern_scale)).trunc() as i64 % 2 == 1
        }));
    } else if (arg.pattern_type == 4) {
        //small dot;
    }

    let bd;
    if (arg.scale_type == 0) {
        bd = todo!("fish_body_a(curve0,curve1,arg.scale_scale,pattern_func)");
    } else if (arg.scale_type == 1) {
        bd = todo!("fish_body_b(curve0,curve1,arg.scale_scale,pattern_func)");
    } else if (arg.scale_type == 2) {
        bd = todo!("fish_body_c(curve0,curve1,arg.scale_scale)");
    } else if (arg.scale_type == 3) {
        bd = todo!("fish_body_d(curve0,curve1,arg.scale_scale)");
    }

    let mut f0_func: Box<dyn Fn(f64) -> f64>;
    let mut f0_a0;
    let mut f0_a1;
    let mut f0_cv;
    if (arg.dorsal_type == 0) {
        f0_a0 = 0.2 + deviate(0.05);
        f0_a1 = 0.3 + deviate(0.05);
        f0_cv = 0.;
        f0_func = Box::new(|t| {
            (0.3 + noise(t * 3., None, None) * 0.7) * arg.dorsal_length * f64::sin(t * PI).powf(0.5)
        });
    } else if (arg.dorsal_type == 1) {
        f0_a0 = 0.6 + deviate(0.05);
        f0_a1 = 0.3 + deviate(0.05);
        f0_cv = arg.dorsal_length / 8.;
        f0_func = Box::new(|t| arg.dorsal_length * ((f64::powi(t - 1., 2)) * 0.5 + (1. - t) * 0.5));
    }
    let mut f0_curve;
    let mut c0: Vec<Polyline>;
    let mut f0: Vec<Polyline>;
    if (arg.dorsal_texture_type == 0) {
        f0_curve = resample(&curve0[arg.dorsal_start..arg.dorsal_end], 5.);
        todo!("let (c0, f0) = fin_a(f0_curve,f0_a0,f0_a1,f0_func,false,f0_cv,0)");
    } else {
        f0_curve = resample(&curve0[arg.dorsal_start..arg.dorsal_end], 15.);
        todo!("let (c0, f0) = fin_b(f0_curve,f0_a0,f0_a1,f0_func)");
    }
    f0 = clip_multi(&f0, &trsl_poly(&outline, 0., 0.001), None).dont_clip;

    let mut f1_curve = vec![];
    let mut f1_func: Box<dyn Fn(f64) -> f64>;
    let mut f1_a0;
    let mut f1_a1;
    let mut f1_soft;
    let mut f1_cv;
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
    if (arg.wing_type == 0) {
        f1_a0 = -0.4 + deviate(0.05);
        f1_a1 = 0.4 + deviate(0.05);
        f1_soft = 10;
        f1_cv = 0.;
        f1_func = Box::new(|t| {
            ((40. + (20. + noise(t * 3., None, None) * 70.) * f64::sin(t * PI).powf(0.5)) / 130.
                * arg.wing_length)
        });
    } else {
        f1_a0 = 0. + deviate(0.05);
        f1_a1 = 0.4 + deviate(0.05);
        f1_soft = 5;
        f1_cv = arg.wing_length / 25.;
        f1_func = Box::new(|t| (arg.wing_length * (1. - t * 0.95)));
    }

    let c1;
    let f1;
    if (arg.wing_texture_type == 0) {
        f1_curve = resample(&f1_curve, 1.5);
        todo!("let (c1, f1) = fin_a(f1_curve,f1_a0,f1_a1,f1_func,1,f1_cv,0,f1_soft)");
    } else {
        f1_curve = resample(&f1_curve, 4.);
        todo!("let (c1, f1) = fin_b(f1_curve,f1_a0,f1_a1,f1_func,0.3)");
    }
    bd = clip_multi(&bd, &c1, None).dont_clip;

    let f2_curve;
    let mut f2_func: Box<dyn Fn(f64) -> f64>;
    let mut f2_a0;
    let mut f2_a1;
    if (arg.pelvic_type == 0) {
        f2_a0 = -0.8 + deviate(0.05);
        f2_a1 = -0.5 + deviate(0.05);
        f2_func = Box::new(|t| {
            (10. + (15. + noise(t * 3., None, None) * 60.) * f64::sin(t * PI).powf(0.5)) / 85.
                * arg.pelvic_length
        });
    } else {
        f2_a0 = -0.9 + deviate(0.05);
        f2_a1 = -0.3 + deviate(0.05);
        f2_func = Box::new(|t| (t * 0.5 + 0.5) * arg.pelvic_length);
    }
    let c2: Vec<Polyline>;
    let f2;
    if (arg.pelvic_texture_type == 0) {
        f2_curve = resample(
            &rev_slice(&curve1, arg.pelvic_start, arg.pelvic_end),
            if arg.pelvic_type != 0 { 2. } else { 5. },
        );
        todo!("let (c2, f2) = fin_a(f2_curve,f2_a0,f2_a1,f2_func)");
    } else {
        f2_curve = resample(
            &rev_slice(&curve1, arg.pelvic_start, arg.pelvic_end),
            if arg.pelvic_type != 0 { 2. } else { 15. },
        );
        todo!("let (c2, f2) = fin_b(f2_curve,f2_a0,f2_a1,f2_func)");
    }
    f2 = clip_multi(&f2, &c1, None).dont_clip;

    let f3_curve;
    let f3_func: Box<dyn Fn(f64) -> f64>;
    let f3_a0;
    let f3_a1;
    if (arg.anal_type == 0) {
        f3_a0 = -0.4 + deviate(0.05);
        f3_a1 = -0.4 + deviate(0.05);
        f3_func = Box::new(|t| {
            (10. + (10. + noise(t * 3., None, None) * 30.) * f64::sin(t * PI).powf(0.5)) / 50.
                * arg.anal_length
        });
    } else {
        f3_a0 = -0.4 + deviate(0.05);
        f3_a1 = -0.4 + deviate(0.05);
        f3_func = Box::new(|t| arg.anal_length * (t * t * 0.8 + 0.2));
    }
    let c3: Vec<Polyline>;
    let f3;
    if (arg.anal_texture_type == 0) {
        f3_curve = resample(&rev_slice(&curve1, arg.anal_start, arg.anal_end), 5.);
        todo!("let (c3, f3) = fin_a(f3_curve,f3_a0,f3_a1,f3_func)");
    } else {
        f3_curve = resample(&rev_slice(&curve1, arg.anal_start, arg.anal_end), 15.);
        todo!("let (c3, f3) = fin_b(f3_curve,f3_a0,f3_a1,f3_func)");
    }
    f3 = clip_multi(&f3, &c1, None).dont_clip;

    let f4_curve;
    let c4;
    let f4;
    let f4_r = dist(curve0[curve0.len() - 2], curve1[curve1.len() - 2]);
    let f4_n = (f4_r / 1.5).trunc();
    f4_n = f64::max(f64::min(f4_n, 20.), 8.);
    let f4_d = f4_r / f4_n;
    // console.log(f4_n,f4_d);
    if (arg.tail_type == 0) {
        f4_curve = vec![curve0[curve0.len() - 1], curve1[curve1.len() - 1]];
        f4_curve = resample(&f4_curve, f4_d);
        todo!("let (c4,f4) = fin_a(f4_curve,-0.6,0.6,t=>(  (75-(10+noise(t*3)*10)*f64::sin(3*t*PI-PI))/75*arg.tail_length  ),1)");
    } else if (arg.tail_type == 1) {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        todo!(
            "let (c4, f4) = fin_a(f4_curve,-0.6,0.6,t=>( arg.tail_length*(f64::sin(t*PI)*0.5+0.5)  ),1)"
        );
    } else if (arg.tail_type == 2) {
        f4_curve = vec![curve0[curve0.len() - 1], curve1[curve1.len() - 1]];
        f4_curve = resample(&f4_curve, f4_d * 0.7);
        let cv = arg.tail_length / 8;
        todo!("let (c4,f4) = fin_a(f4_curve,-0.6,0.6,t=>(  (Math.abs(Math.cos(PI*t))*0.8+0.2)*arg.tail_length  ),1,cv,-cv)");
    } else if (arg.tail_type == 3) {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        todo!(
            "let (c4, f4) = fin_a(f4_curve,-0.6,0.6,t=>(  (1-f64::sin(t*PI)*0.3)*arg.tail_length  ),1)"
        );
    } else if (arg.tail_type == 4) {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        todo!("let (c4, f4) = fin_a(f4_curve,-0.6,0.6,t=>(  (1-f64::sin(t*PI)*0.6)*(1-t*0.45)*arg.tail_length  ),1)"
        );
    } else if (arg.tail_type == 5) {
        f4_curve = vec![curve0[curve0.len() - 2], curve1[curve1.len() - 2]];
        f4_curve = resample(&f4_curve, f4_d);
        todo!("let (c4, f4) = fin_a(f4_curve,-0.6,0.6,t=>(  (1-f64::sin(t*PI)**0.4*0.55)*arg.tail_length  ),1)"
        );
    }
    // f4 = clip_multi(f4,trsl_poly(outline,-1,0)).dont_clip;
    bd = clip_multi(&bd, &trsl_poly(c4, 1., 0.), None).dont_clip;

    f4 = clip_multi(&f4, &c1, None).dont_clip;

    let f5_curve = vec![];
    let c5 = vec![];
    let f5 = vec![];
    if (arg.finlet_type == 0) {
        //pass
    } else if (arg.finlet_type == 1) {
        f5_curve = resample(&curve0[arg.dorsal_end..curve0.len() - 2], 5.);
        todo!("let (c5, f5) = finlet(f5_curve,5)");
        f5_curve = resample(&rev_slice(&curve1, arg.anal_end, curve1.len() - 2), 5.);
        if (f5_curve.len() > 1) {
            todo!("let (c5, f5) = finlet(f5_curve,5)");
        }
    } else if (arg.finlet_type == 2) {
        f5_curve = resample(&curve0[27..30], 5.);
        todo!("let (c5, f5) = fin_adipose(f5_curve,20,-5,6)");
        outline = poly_union(&outline, &trsl_poly(&c5, 0., -1.), None);
    } else {
        f5_curve = resample(&curve0[arg.dorsal_end + 2..curve0.len() - 3], 5.);
        if (f5_curve.len() > 2) {
            todo!("let (c5,f5) = fin_a(f5_curve,0.2,0.3, t=>(  (0.3+noise(t*3)*0.7)*arg.dorsal_length*0.6 *f64::sin(t*PI)**0.5  ))");
        }
    }
    let cf;
    let fh: Vec<Polyline>;
    if (arg.neck_type == 0) {
        todo!("let (cf, fh) = fish_head(50-arg.head_length,150+arg.nose_height,...curve0[6],...curve1[5],arg)"
        );
    } else {
        todo!("let (cf, fh) = fish_head(50-arg.head_length,150+arg.nose_height,...curve0[5],...curve1[6],arg)"
        );
    }
    bd = clip_multi(&bd, &cf, None).dont_clip;

    sh = clip_multi(&sh, &cf, None).dont_clip;
    sh = clip_multi(&sh, &c1, None).dont_clip;

    f1 = clip_multi(&f1, &cf, None).dont_clip;

    f0 = clip_multi(&f0, &c1, None).dont_clip;

    let sh2 = vec![];
    if let Some(func) = (pattern_func) {
        if (arg.scale_type > 1) {
            sh2 = todo!(
                "patternshade_shape(&poly_union(outline, &trsl_poly(c0, 0., 3.), None),3.5,func)"
            );
        } else {
            sh2 = todo!("patternshade_shape(c0, 4.5, func)");
        }
        sh2 = clip_multi(&sh2, &cf, None).dont_clip;
        sh2 = clip_multi(&sh2, &c1, None).dont_clip;
    }

    let sh3 = [];
    if (arg.pattern_type == 4) {
        sh3 = todo!("smalldot_shape(poly_union(outline, trsl_poly(c0, 0, 5)), arg.pattern_scale)");
        sh3 = todo!("clip_multi(&sh3, &c1, None).dont_clip");
        sh3 = todo!("clip_multi(&sh3, &cf, None).dont_clip");
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
    // bd.extend([cf]);
    bd
}

fn put_text(txt: String) -> (f64, Vec<Vec<(f64, f64)>>) {
    let base = 500;
    let mut x = 0.;
    let mut o = vec![];
    for c in txt.chars() {
        let ord = c as i32;
        let idx;
        if (65 <= ord && ord <= 90) {
            idx = base + 1 + (ord - 65);
        } else if (97 <= ord && ord <= 122) {
            idx = base + 101 + (ord - 97);
        } else if (ord == 46) {
            idx = 710;
        } else if (ord == 32) {
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

    let w = (500. - pad * 2.);
    let h = ((300. - pad * 2.) - (if text.is_some() { 10. } else { 0. }));
    let bbox = get_boundingbox(&polylines.clone().into_iter().flatten().collect());
    let sw = w / bbox.w;
    let sh = h as f64 / bbox.h;
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
    let (mut tw, mut tp) = put_text(text.unwrap_or(String::new()));
    todo!();
    // tp = tp.into_iter().map(|p| scl_poly(shr_poly(p,-0.3),0.3,0.3));
    tw *= 0.3;
    polylines.extend(
        tp.into_iter()
            .map(|p| trsl_poly(&p, 250. - tw / 2., 300. - pad + 5.)),
    );
    return polylines;
}

pub fn cleanup(polylines: Vec<Polyline>) -> Vec<Vec<(f64, f64)>> {
    for i in (polylines.len() - 1)..=0 {
        polylines[i] = todo!(); //approx_poly_dp(polylines[i], 0.1);
        for j in 0..polylines[i].len() {
            polylines[i][j] = (
                (polylines[i][j].0 * 10000.).trunc() / 10000.,
                (polylines[i][j].1 * 10000.).trunc() / 10000.,
            );
        }
        if (polylines[i].len() < 2) {
            polylines.splice(i..1, []);
            continue;
        }
        if (polylines[i].len() == 2) {
            if (dist(polylines[i][0], polylines[i][1]) < 0.9) {
                polylines.splice(i..1, []);
                continue;
            }
        }
    }
    return polylines;
}
