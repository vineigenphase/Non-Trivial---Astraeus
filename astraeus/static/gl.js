/* Astraeus world view: raw WebGL, no dependencies.
 * Draws the simulator's own height field (the DEM the episode was run on),
 * rocks, the rover at the scrubbed frame, true / estimated trajectories and
 * the goal ring. Lighting is the same Lommel-Seeliger regolith model as the
 * software renderer, with a per-vertex terrain shadow march toward the sun. */
"use strict";

const M4 = {
  ident() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+j] * b[i*4+k]; o[i*4+j] = s;
    }
    return o;
  },
  persp(fovy, aspect, n, f) {
    const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = t / aspect; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f);
    return o;
  },
  lookAt(eye, at, up) {
    const z = norm3(sub3(eye, at)), x = norm3(cross3(up, z)), y = cross3(z, x);
    return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
      -dot3(x,eye), -dot3(y,eye), -dot3(z,eye), 1]);
  },
  translate(v) { const o = M4.ident(); o[12] = v[0]; o[13] = v[1]; o[14] = v[2]; return o; },
  scale(v) { const o = M4.ident(); o[0] = v[0]; o[5] = v[1]; o[10] = v[2]; return o; },
  /* body->world: yaw about z, pitch about y (nose-up +), roll about x (right-down +) */
  pose(yaw, pitch, roll) {
    const ch = Math.cos(yaw), sh = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch), cr = Math.cos(roll), sr = Math.sin(roll);
    const Rz = new Float32Array([ch,sh,0,0, -sh,ch,0,0, 0,0,1,0, 0,0,0,1]);
    const Ry = new Float32Array([cp,0,sp,0, 0,1,0,0, -sp,0,cp,0, 0,0,0,1]);
    const Rx = new Float32Array([1,0,0,0, 0,cr,sr,0, 0,-sr,cr,0, 0,0,0,1]);
    return M4.mul(Rz, M4.mul(Ry, Rx));
  },
};
function sub3(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function dot3(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross3(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm3(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

const VS = `
attribute vec3 aPos; attribute vec3 aNrm; attribute float aShade;
uniform mat4 uP, uV, uM; uniform vec3 uSun; uniform vec3 uEye;
varying vec3 vN; varying vec3 vW; varying float vSh;
void main(){ vec4 w = uM * vec4(aPos,1.0); vW = w.xyz; vN = normalize(mat3(uM) * aNrm); vSh = aShade;
  gl_Position = uP * uV * w; }`;
const FS = `
precision mediump float;
varying vec3 vN; varying vec3 vW; varying float vSh;
uniform vec3 uSun; uniform vec3 uEye; uniform vec3 uColor; uniform float uEmissive; uniform float uTau; uniform int uMode;
void main(){
  vec3 n = normalize(vN); vec3 v = normalize(uEye - vW);
  float mu0 = max(dot(n, uSun), 0.0); float mu = max(dot(n, v), 0.0);
  float phase = acos(clamp(dot(v, uSun), -1.0, 1.0));
  float ls = mu0 / max(mu0 + mu, 1e-3);
  float surge = 1.0 + 0.6 * exp(-phase / 0.105);
  float amb = 0.03 + 0.25 * uTau;
  vec3 col;
  if (uMode == 0) {            // regolith: Lommel-Seeliger, per-vertex shadow
    float grain = 0.92 + 0.16 * fract(sin(dot(floor(vW.xy * 4.0), vec2(12.9898, 78.233))) * 43758.5453);
    col = uColor * grain * (2.2 * 0.11 * ls * surge * vSh + amb * 0.3);
  } else if (uMode == 1) {     // rover materials: lambert + spec
    vec3 h = normalize(uSun + v); float spec = pow(max(dot(n, h), 0.0), 30.0);
    col = uColor * (0.9 * mu0 * vSh + amb) + 0.35 * spec * vSh;
  } else {                     // emissive markers
    col = uColor * uEmissive;
  }
  float T = exp(-uTau * length(uEye - vW) / 40.0);
  col = col * T + (1.0 - T) * uTau * 0.3 * vec3(0.86, 0.80, 0.70);
  float ex = 1.3 / max(0.35, sqrt(max(uSun.z, 0.02)) * 2.2);
  col *= ex; col = 1.0 - exp(-1.6 * col); col = pow(col, vec3(1.0/2.2));
  gl_FragColor = vec4(col, 1.0);
}`;

class World {
  constructor(canvas) {
    this.cv = canvas;
    const gl = this.gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: false });
    if (!gl) throw new Error("WebGL unavailable");
    this.prog = this._prog(VS, FS);
    this.loc = {};
    for (const n of ["uP","uV","uM","uSun","uEye","uColor","uEmissive","uTau","uMode"]) this.loc[n] = gl.getUniformLocation(this.prog, n);
    this.attr = { aPos: gl.getAttribLocation(this.prog, "aPos"), aNrm: gl.getAttribLocation(this.prog, "aNrm"), aShade: gl.getAttribLocation(this.prog, "aShade") };
    this.cube = this._mesh(World.cube());
    this.sphere = this._mesh(World.sphere(10, 7));
    this.cyl = this._mesh(World.cylinder(18));
    this.terrain = null; this.paths = { t: null, e: null }; this.ring = null;
    this.world = null; this.frame = 0;
    this.cam = { yaw: -2.35, pitch: 0.55, dist: 26, target: [15, 0, 0], follow: false };
    this.showEst = true; this.showTrue = true;
    this._mouse();
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
  }

  /* ---------------------------------------------------------------- data */
  setWorld(w) {
    this.world = w; this.frame = w.n_frames - 1;
    this._buildTerrain(w);
    this._buildPaths(w);
    const gl = this.gl;
    // goal ring
    const pts = [];
    for (let i = 0; i <= 64; i++) { const a = i / 64 * Math.PI * 2; const x = w.goal.x + w.goal.tol * Math.cos(a), y = w.goal.tol * Math.sin(a); pts.push(x, y, this.height(x, y) + 0.05); }
    this.ring = this._line(new Float32Array(pts));
    if (!this.cam.follow) this.cam.target = [w.goal.x * 0.5, 0, this.height(w.goal.x * 0.5, 0)];
    this.draw();
  }
  setFrame(k) { this.frame = Math.max(0, Math.min(k, this.world.n_frames - 1)); this._buildPaths(this.world, this.frame); this.draw(); }

  height(x, y) {
    const d = this.world.dem;
    const fx = (x - d.x0) / d.res, fy = (y - d.y0) / d.res;
    const ix = Math.max(0, Math.min(d.nx - 2, Math.floor(fx))), iy = Math.max(0, Math.min(d.ny - 2, Math.floor(fy)));
    const tx = Math.max(0, Math.min(1, fx - ix)), ty = Math.max(0, Math.min(1, fy - iy));
    const z = d.z, i = iy * d.nx + ix;
    return (z[i] * (1 - tx) + z[i + 1] * tx) * (1 - ty) + (z[i + d.nx] * (1 - tx) + z[i + d.nx + 1] * tx) * ty;
  }

  _buildTerrain(w) {
    const d = w.dem, nx = d.nx, ny = d.ny, z = d.z;
    const pos = new Float32Array(nx * ny * 3), nrm = new Float32Array(nx * ny * 3), sh = new Float32Array(nx * ny);
    const sun = w.sun;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i, x = d.x0 + i * d.res, y = d.y0 + j * d.res;
      pos[3*k] = x; pos[3*k+1] = y; pos[3*k+2] = z[k];
      const zl = z[j*nx + Math.max(i-1,0)], zr = z[j*nx + Math.min(i+1,nx-1)];
      const zd = z[Math.max(j-1,0)*nx + i], zu = z[Math.min(j+1,ny-1)*nx + i];
      const n = norm3([-(zr - zl) / (2 * d.res), -(zu - zd) / (2 * d.res), 1]);
      nrm[3*k] = n[0]; nrm[3*k+1] = n[1]; nrm[3*k+2] = n[2];
      // shadow march toward the sun (sun.z is tiny at the pole, so this matters)
      let vis = 1, t = 0.4;
      for (let s = 0; s < 60 && vis > 0; s++) {
        const px = x + t * sun[0], py = y + t * sun[1], pz = z[k] + 0.03 + t * sun[2];
        if (px < d.x0 || py < d.y0 || px > d.x0 + (nx - 1) * d.res || py > d.y0 + (ny - 1) * d.res) break;
        const h = this._h(d, px, py);
        const dh = pz - h;
        vis = Math.min(vis, Math.max(0, Math.min(1, 10 * dh / t)));
        t += Math.max(0.15, 0.6 * Math.abs(dh));
      }
      sh[k] = vis;
    }
    const idx = [];
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, e = c + 1;
      idx.push(a, b, e, a, e, c);
    }
    this.terrain = this._mesh({ pos, nrm, shade: sh, idx: new Uint16Array(idx) }, true);
    this.rocks = w.rocks.map(r => ({ x: r.x, y: r.y, r: r.r, z: this._h(d, r.x, r.y) - 0.35 * r.r }));
  }
  _h(d, x, y) {
    const fx = (x - d.x0) / d.res, fy = (y - d.y0) / d.res;
    const ix = Math.max(0, Math.min(d.nx - 2, Math.floor(fx))), iy = Math.max(0, Math.min(d.ny - 2, Math.floor(fy)));
    const tx = Math.max(0, Math.min(1, fx - ix)), ty = Math.max(0, Math.min(1, fy - iy));
    const z = d.z, i = iy * d.nx + ix;
    return (z[i] * (1 - tx) + z[i + 1] * tx) * (1 - ty) + (z[i + d.nx] * (1 - tx) + z[i + d.nx + 1] * tx) * ty;
  }
  _buildPaths(w, upto) {
    const k = upto === undefined ? w.n_frames - 1 : upto, tr = w.trace;
    const pt = [], pe = [];
    for (let i = 0; i <= k; i++) {
      pt.push(tr.x[i], tr.y[i], tr.z[i] + 0.08);
      pe.push(tr.ex[i], tr.ey[i], this.height(tr.ex[i], tr.ey[i]) + 0.12);
    }
    this.paths.t = this._line(new Float32Array(pt));
    this.paths.e = this._line(new Float32Array(pe));
  }

  /* ------------------------------------------------------------ drawing */
  draw() {
    const gl = this.gl, cv = this.cv, w = this.world;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(cv.clientWidth * dpr), H = Math.floor(cv.clientHeight * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!w) return;
    const k = this.frame, tr = w.trace;
    const rx = tr.x[k], ry = tr.y[k], rz = tr.z[k];
    const tgt = this.cam.follow ? [rx, ry, rz + 0.5] : this.cam.target;
    const c = this.cam;
    const eye = [tgt[0] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw), tgt[1] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw), tgt[2] + c.dist * Math.sin(c.pitch)];
    const P = M4.persp(0.9, W / H, 0.1, 400), V = M4.lookAt(eye, tgt, [0, 0, 1]);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.loc.uP, false, P); gl.uniformMatrix4fv(this.loc.uV, false, V);
    gl.uniform3fv(this.loc.uSun, w.sun); gl.uniform3fv(this.loc.uEye, eye);
    gl.uniform1f(this.loc.uTau, w.x.tau_dust);

    // terrain
    this._drawMesh(this.terrain, M4.ident(), [1.0, 0.97, 0.92], 0, 0);
    // rocks (buried 35 %)
    for (const r of this.rocks) {
      const M = M4.mul(M4.translate([r.x, r.y, r.z]), M4.scale([r.r, r.r, r.r * 0.8]));
      this._drawMesh(this.sphere, M, [0.95, 0.93, 0.9], 0, 0);
    }
    // rover
    const pose = M4.mul(M4.translate([rx, ry, rz + w.rover.wheel_r]), M4.pose(tr.heading[k], tr.pitch[k], tr.roll[k]));
    const part = (mesh, off, half, col, mode) => this._drawMesh(mesh, M4.mul(pose, M4.mul(M4.translate(off), M4.scale(half))), col, 0, mode === undefined ? 1 : mode);
    part(this.cube, [0, 0, 0.54], [0.85, 0.62, 0.22], [0.85, 0.80, 0.62]);
    part(this.cube, [-0.2, 0, 0.90], [0.5, 0.55, 0.14], [0.85, 0.80, 0.62]);
    part(this.cube, [0.55, 0, 1.45], [0.05, 0.05, 0.45], [0.22, 0.22, 0.24]);
    part(this.cube, [0.55, 0, 1.95], [0.16, 0.22, 0.08], [0.22, 0.22, 0.24]);
    part(this.cube, [-0.55, 0, 1.20], [0.32, 0.65, 0.03], [0.10, 0.12, 0.28]);
    for (const sx of [w.rover.half_len, -w.rover.half_len]) for (const sy of [w.rover.half_wid, -w.rover.half_wid]) {
      part(this.cyl, [sx, sy, 0], [w.rover.wheel_r, 0.10, w.rover.wheel_r], [0.30, 0.30, 0.30]);
      part(this.cube, [sx, sy * 0.82, 0.25], [0.06, 0.18, 0.28], [0.22, 0.22, 0.24]);
    }
    // overlays
    gl.disable(gl.DEPTH_TEST);
    if (this.showTrue) this._drawLine(this.paths.t, [0.10, 0.75, 0.95], 1.0);
    if (this.showEst) this._drawLine(this.paths.e, [1.0, 0.55, 0.10], 1.0);
    this._drawLine(this.ring, [0.35, 1.0, 0.45], 1.0, true);
    // sun direction indicator from the rover
    const sl = this._line(new Float32Array([rx, ry, rz + 0.3, rx + 4 * w.sun[0], ry + 4 * w.sun[1], rz + 0.3 + 4 * w.sun[2]]));
    this._drawLine(sl, [1.0, 0.9, 0.6], 0.7);
    gl.deleteBuffer(sl.buf);
    gl.enable(gl.DEPTH_TEST);
  }

  _drawMesh(m, M, col, em, mode) {
    const gl = this.gl;
    gl.uniformMatrix4fv(this.loc.uM, false, M); gl.uniform3fv(this.loc.uColor, col);
    gl.uniform1f(this.loc.uEmissive, em); gl.uniform1i(this.loc.uMode, mode);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.pos); gl.enableVertexAttribArray(this.attr.aPos); gl.vertexAttribPointer(this.attr.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.nrm); gl.enableVertexAttribArray(this.attr.aNrm); gl.vertexAttribPointer(this.attr.aNrm, 3, gl.FLOAT, false, 0, 0);
    if (m.shade) { gl.bindBuffer(gl.ARRAY_BUFFER, m.shade); gl.enableVertexAttribArray(this.attr.aShade); gl.vertexAttribPointer(this.attr.aShade, 1, gl.FLOAT, false, 0, 0); }
    else { gl.disableVertexAttribArray(this.attr.aShade); gl.vertexAttrib1f(this.attr.aShade, 1.0); }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx); gl.drawElements(gl.TRIANGLES, m.n, m.type, 0);
  }
  _drawLine(l, col, em, loop) {
    const gl = this.gl;
    if (!l || l.n < 2) return;
    gl.uniformMatrix4fv(this.loc.uM, false, M4.ident()); gl.uniform3fv(this.loc.uColor, col);
    gl.uniform1f(this.loc.uEmissive, em); gl.uniform1i(this.loc.uMode, 2);
    gl.bindBuffer(gl.ARRAY_BUFFER, l.buf); gl.enableVertexAttribArray(this.attr.aPos); gl.vertexAttribPointer(this.attr.aPos, 3, gl.FLOAT, false, 0, 0);
    gl.disableVertexAttribArray(this.attr.aNrm); gl.vertexAttrib3f(this.attr.aNrm, 0, 0, 1);
    gl.disableVertexAttribArray(this.attr.aShade); gl.vertexAttrib1f(this.attr.aShade, 1);
    gl.drawArrays(loop ? gl.LINE_LOOP : gl.LINE_STRIP, 0, l.n);
  }

  /* ----------------------------------------------------------- buffers */
  _prog(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [t, s] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
      gl.attachShader(p, sh);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  _buf(target, data) { const gl = this.gl, b = gl.createBuffer(); gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW); return b; }
  _mesh(m, replace) {
    const gl = this.gl;
    if (replace && this.terrain) for (const k of ["pos", "nrm", "shade", "idx"]) if (this.terrain[k]) gl.deleteBuffer(this.terrain[k]);
    const big = m.idx.length > 65535 || m.pos.length / 3 > 65535;
    const idx = big ? new Uint32Array(m.idx) : new Uint16Array(m.idx);
    if (big) gl.getExtension("OES_element_index_uint");
    return { pos: this._buf(gl.ARRAY_BUFFER, m.pos), nrm: this._buf(gl.ARRAY_BUFFER, m.nrm),
      shade: m.shade ? this._buf(gl.ARRAY_BUFFER, m.shade) : null,
      idx: this._buf(gl.ELEMENT_ARRAY_BUFFER, idx), n: idx.length, type: big ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT };
  }
  _line(arr) {
    const gl = this.gl;
    return { buf: this._buf(gl.ARRAY_BUFFER, arr), n: arr.length / 3 };
  }

  /* ------------------------------------------------------------- input */
  _mouse() {
    const cv = this.cv; let drag = null;
    cv.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, b: e.button }; e.preventDefault(); });
    window.addEventListener("mouseup", () => drag = null);
    window.addEventListener("mousemove", e => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (drag.b === 2 || e.shiftKey) {
        const c = this.cam, s = c.dist * 0.0018;
        const rgt = [-Math.sin(c.yaw), Math.cos(c.yaw), 0];
        c.target = [c.target[0] - rgt[0] * dx * s - Math.cos(c.yaw) * dy * s, c.target[1] - rgt[1] * dx * s - Math.sin(c.yaw) * dy * s, c.target[2]];
        c.follow = false;
      } else {
        this.cam.yaw -= dx * 0.006; this.cam.pitch = Math.max(0.05, Math.min(1.5, this.cam.pitch + dy * 0.006));
      }
      this.draw();
    });
    cv.addEventListener("wheel", e => { this.cam.dist = Math.max(3, Math.min(120, this.cam.dist * Math.exp(e.deltaY * 0.001))); this.draw(); e.preventDefault(); }, { passive: false });
    cv.addEventListener("contextmenu", e => e.preventDefault());
    window.addEventListener("resize", () => this.draw());
  }

  /* --------------------------------------------------------- primitives */
  static cube() {
    const p = [], n = [], idx = [];
    const faces = [[[1,0,0],[0,1,0],[0,0,1]], [[-1,0,0],[0,0,1],[0,1,0]], [[0,1,0],[0,0,1],[1,0,0]], [[0,-1,0],[1,0,0],[0,0,1]], [[0,0,1],[1,0,0],[0,1,0]], [[0,0,-1],[0,1,0],[1,0,0]]];
    for (const [nn, u, v] of faces) {
      const b = p.length / 3;
      for (const [su, sv] of [[-1,-1],[1,-1],[1,1],[-1,1]]) {
        p.push(nn[0] + su*u[0] + sv*v[0], nn[1] + su*u[1] + sv*v[1], nn[2] + su*u[2] + sv*v[2]); n.push(...nn);
      }
      idx.push(b, b+1, b+2, b, b+2, b+3);
    }
    return { pos: new Float32Array(p), nrm: new Float32Array(n), idx: new Uint16Array(idx) };
  }
  static sphere(seg, rings) {
    const p = [], n = [], idx = [];
    for (let j = 0; j <= rings; j++) { const th = Math.PI * j / rings; for (let i = 0; i <= seg; i++) { const ph = 2 * Math.PI * i / seg;
      const v = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]; p.push(...v); n.push(...v); } }
    for (let j = 0; j < rings; j++) for (let i = 0; i < seg; i++) { const a = j * (seg + 1) + i, b = a + seg + 1; idx.push(a, b, a + 1, a + 1, b, b + 1); }
    return { pos: new Float32Array(p), nrm: new Float32Array(n), idx: new Uint16Array(idx) };
  }
  static cylinder(seg) {      // axis along y, radius 1, half-width 1
    const p = [], n = [], idx = [];
    for (let i = 0; i <= seg; i++) { const a = 2 * Math.PI * i / seg, c = Math.cos(a), s = Math.sin(a);
      p.push(c, -1, s, c, 1, s); n.push(c, 0, s, c, 0, s); }
    for (let i = 0; i < seg; i++) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    for (const sy of [-1, 1]) { const b = p.length / 3; p.push(0, sy, 0); n.push(0, sy, 0);
      for (let i = 0; i <= seg; i++) { const a = 2 * Math.PI * i / seg; p.push(Math.cos(a), sy, Math.sin(a)); n.push(0, sy, 0); }
      for (let i = 0; i < seg; i++) { if (sy > 0) idx.push(b, b + 2 + i, b + 1 + i); else idx.push(b, b + 1 + i, b + 2 + i); } }
    return { pos: new Float32Array(p), nrm: new Float32Array(n), idx: new Uint16Array(idx) };
  }
}
window.AstraeusWorld = World;
