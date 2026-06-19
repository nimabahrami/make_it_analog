// =============================================================
//  Halation Filter — real-time WebGL2 engine
//  Every effect runs on the GPU so previews are live & instant.
// =============================================================

const canvas = document.getElementById("gl-canvas");
const gl = canvas.getContext("webgl2", {
  premultipliedAlpha: false,
  preserveDrawingBuffer: true, // needed for toBlob() download
  antialias: false,
});

function fatal(msg) {
  const note = document.getElementById("error-notification");
  document.getElementById("error-message").innerText = msg;
  note.style.display = "block";
  console.error(msg);
}

if (!gl) {
  fatal("Your browser does not support WebGL2, which this filter requires.");
}

// ---------------------------------------------------------------
//  Shader sources
// ---------------------------------------------------------------

const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main(){
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const COMMON = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 frag;
vec3 toLinear(vec3 c){ return pow(max(c, 0.0), vec3(2.2)); }
vec3 toSRGB(vec3 c){ return pow(max(c, 0.0), vec3(1.0/2.2)); }
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
`;

// Tone curve — luminance-zone weighted shadows / mids / highlights.
const FS_TONE = COMMON + `
uniform sampler2D u_src;
uniform float u_shadows, u_midtones, u_highlights;
void main(){
  vec3 lin = toLinear(texture(u_src, v_uv).rgb);
  if(u_shadows != 0.0 || u_midtones != 0.0 || u_highlights != 0.0){
    float y = dot(lin, LUMA);
    float sw = clamp(1.0 - y * 3.0, 0.0, 1.0); sw *= sw;
    float hw = clamp((y - 0.6) * 2.5, 0.0, 1.0); hw *= hw;
    float mw = clamp(1.0 - sw - hw, 0.0, 1.0);
    float adj = sw*u_shadows*0.3 + mw*u_midtones*0.3 + hw*u_highlights*0.3;
    lin = max(lin + adj, 0.0);
  }
  frag = vec4(toSRGB(lin), 1.0);
}`;

// Bright-pass extraction (used for halation, bloom and streaks).
const FS_BRIGHT = COMMON + `
uniform sampler2D u_src;
uniform float u_threshold, u_softness;
void main(){
  vec3 lin = toLinear(texture(u_src, v_uv).rgb);
  float bright = max(lin.r, max(lin.g, lin.b));
  float m = clamp((bright - u_threshold) / max(1e-4, 1.0 - u_threshold), 0.0, 1.0);
  m = pow(m, u_softness);
  frag = vec4(toSRGB(lin * m), 1.0);
}`;

// Separable Gaussian (9 taps). Run twice per axis for a wide, smooth glow.
const FS_BLUR = COMMON + `
uniform sampler2D u_src;
uniform vec2 u_dir;       // (1,0) or (0,1)
uniform vec2 u_texel;     // 1/size
uniform float u_spacing;  // tap spacing in texels
void main(){
  const float W[9] = float[](0.0276,0.0663,0.1238,0.1797,0.2042,0.1797,0.1238,0.0663,0.0276);
  vec2 step = u_dir * u_texel * u_spacing;
  vec3 acc = vec3(0.0);
  for(int i=0;i<9;i++){
    float o = float(i - 4);
    acc += texture(u_src, v_uv + step * o).rgb * W[i];
  }
  frag = vec4(acc, 1.0);
}`;

// Directional streak blur — long one-dimensional smear with falloff.
const FS_STREAK = COMMON + `
uniform sampler2D u_src;
uniform vec2 u_dir;       // unit direction
uniform vec2 u_texel;
uniform float u_spacing;
void main(){
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for(int i=-14;i<=14;i++){
    float fi = float(i);
    float w = exp(-abs(fi) * 0.16);
    acc += texture(u_src, v_uv + u_dir * u_texel * u_spacing * fi).rgb * w;
    wsum += w;
  }
  frag = vec4(acc / wsum, 1.0);
}`;

// Final composite — everything combined in one pass.
const FS_COMPOSITE = COMMON + `
uniform sampler2D u_base;   // tone-curved image (sRGB)
uniform sampler2D u_src;    // original image (sRGB) for fade
uniform sampler2D u_hal;    // halation glow
uniform sampler2D u_bloom;  // bloom glow
uniform sampler2D u_streak; // streaks

uniform float u_halStrength;
uniform vec3  u_halTint;
uniform float u_bloomStrength;
uniform vec3  u_bloomTint;
uniform float u_streakStrength;
uniform float u_aberration;
uniform float u_grain;
uniform float u_grainColor;
uniform float u_mute;
uniform float u_fade;
uniform float u_seed;

uniform float u_leakStrength;
uniform vec3  u_leakColor;
uniform vec2  u_leakPos;

float hash(vec2 p){
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec3 lightLeak(vec2 uv){
  if(u_leakStrength <= 0.0) return vec3(0.0);
  float d = distance(uv, u_leakPos);
  float core = smoothstep(1.05, 0.0, d);
  float hot  = smoothstep(0.55, 0.0, d);
  float band = 0.6 + 0.4 * sin(d * 16.0 - 1.0);
  float inten = (core * 0.55 + hot * 0.8) * band;
  return toLinear(u_leakColor) * inten * u_leakStrength;
}

void main(){
  vec2 center = vec2(0.5);
  vec2 dir = v_uv - center;

  // --- Chromatic aberration: radial RGB split ---
  float ca = u_aberration * 0.02;
  vec3 lin;
  lin.r = toLinear(texture(u_base, v_uv + dir * ca).rgb).r;
  lin.g = toLinear(texture(u_base, v_uv).rgb).g;
  lin.b = toLinear(texture(u_base, v_uv - dir * ca).rgb).b;

  // --- Halation ---
  if(u_halStrength > 0.0){
    vec3 h = toLinear(texture(u_hal, v_uv).rgb);
    float hl = (h.r + h.g + h.b) / 3.0;
    vec3 layer = h * 0.35 + hl * u_halTint * 0.65;
    lin += layer * u_halStrength;
  }

  // --- Golden bloom ---
  if(u_bloomStrength > 0.0){
    vec3 b = toLinear(texture(u_bloom, v_uv).rgb);
    float bl = (b.r + b.g + b.b) / 3.0;
    lin += bl * u_bloomTint * u_bloomStrength;
  }

  // --- Anamorphic streaks ---
  if(u_streakStrength > 0.0){
    lin += toLinear(texture(u_streak, v_uv).rgb) * u_streakStrength;
  }

  // --- Light leaks (screen blend) ---
  vec3 leak = lightLeak(v_uv);
  if(u_leakStrength > 0.0){
    vec3 c = clamp(lin, 0.0, 1.0);
    lin = 1.0 - (1.0 - c) * (1.0 - clamp(leak, 0.0, 1.0));
  }

  // --- Grain (luminance weighted toward shadows) ---
  if(u_grain > 0.0){
    float n = hash(gl_FragCoord.xy + u_seed) - 0.5;
    float yv = dot(clamp(lin, 0.0, 1.0), LUMA);
    float w = pow(1.0 - clamp(yv, 0.0, 1.0), 0.6);
    lin += n * w * u_grain;
    if(u_grainColor > 0.0){
      vec3 cn = vec3(hash(gl_FragCoord.xy + u_seed + 11.0),
                     hash(gl_FragCoord.xy + u_seed + 23.0),
                     hash(gl_FragCoord.xy + u_seed + 37.0)) - 0.5;
      cn -= (cn.r + cn.g + cn.b) / 3.0;
      lin += cn * w * u_grain * u_grainColor;
    }
  }

  // --- Mute (desaturate) ---
  vec3 c = clamp(lin, 0.0, 1.0);
  if(u_mute > 0.0){
    float lum = dot(c, LUMA);
    c = mix(c, vec3(lum), u_mute);
  }

  vec3 srgb = toSRGB(c);

  // --- Fade toward original ---
  if(u_fade > 0.0){
    vec3 orig = texture(u_src, v_uv).rgb;
    srgb = mix(srgb, orig, u_fade);
  }

  frag = vec4(srgb, 1.0);
}`;

// ---------------------------------------------------------------
//  GL helpers
// ---------------------------------------------------------------

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
  }
  return s;
}

function program(fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "a_pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  // Cache uniform locations lazily.
  const locs = {};
  p.u = (name) => (locs[name] ??= gl.getUniformLocation(p, name));
  return p;
}

let progs = null;
function buildPrograms() {
  progs = {
    tone: program(FS_TONE),
    bright: program(FS_BRIGHT),
    blur: program(FS_BLUR),
    streak: program(FS_STREAK),
    composite: program(FS_COMPOSITE),
  };
}

// Fullscreen quad (triangle strip).
let quadVAO;
function buildQuad() {
  quadVAO = gl.createVertexArray();
  gl.bindVertexArray(quadVAO);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
    gl.STATIC_DRAW
  );
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
}

function makeTarget(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return { tex, fb, w, h };
}

function bindTarget(t) {
  if (t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    gl.viewport(0, 0, t.w, t.h);
  } else {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
}

function drawQuad() {
  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function useTex(unit, tex) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
}

// ---------------------------------------------------------------
//  Image source + render targets
// ---------------------------------------------------------------

let srcTex = null;
let imgW = 0, imgH = 0;       // full (capped) image size
let targets = null;            // allocated framebuffers for current canvas size

const MAX_FULL = 2500;         // hard cap on processed resolution (download)
const MAX_PREVIEW = 1600;      // live preview resolution

function uploadImage(img) {
  let w = img.naturalWidth, h = img.naturalHeight;
  // Cap the working resolution.
  if (w > MAX_FULL || h > MAX_FULL) {
    const s = MAX_FULL / Math.max(w, h);
    w = Math.round(w * s);
    h = Math.round(h * s);
  }
  imgW = w;
  imgH = h;

  if (!srcTex) srcTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, srcTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  // Draw the image to a sizing canvas so it matches the capped resolution.
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function freeTargets() {
  if (!targets) return;
  for (const k in targets) {
    const t = targets[k];
    gl.deleteTexture(t.tex);
    gl.deleteFramebuffer(t.fb);
  }
  targets = null;
}

// Allocate all framebuffers for a given canvas size.
function allocTargets(w, h) {
  freeTargets();
  canvas.width = w;
  canvas.height = h;
  const gw = Math.max(1, Math.round(w / 4));
  const gh = Math.max(1, Math.round(h / 4));
  const sw = Math.max(1, Math.round(w / 2));
  const sh = Math.max(1, Math.round(h / 2));
  targets = {
    base: makeTarget(w, h),
    halA: makeTarget(gw, gh),
    halB: makeTarget(gw, gh),
    bloomA: makeTarget(gw, gh),
    bloomB: makeTarget(gw, gh),
    streakBright: makeTarget(sw, sh),
    streakA: makeTarget(sw, sh),
    streakB: makeTarget(sw, sh),
  };
}

// Fit (w,h) within a max longest-side, returning integer dimensions.
function fitTo(w, h, max) {
  if (w <= max && h <= max) return [w, h];
  const s = max / Math.max(w, h);
  return [Math.round(w * s), Math.round(h * s)];
}

// ---------------------------------------------------------------
//  Parameters (read live from the DOM)
// ---------------------------------------------------------------

function val(id) { return parseFloat(document.getElementById(id).value); }
function hexToRGB(hex) {
  hex = hex.replace("#", "");
  return [
    parseInt(hex.slice(0, 2), 16) / 255,
    parseInt(hex.slice(2, 4), 16) / 255,
    parseInt(hex.slice(4, 6), 16) / 255,
  ];
}

const HAL_TINT = [1.0, 0.45, 0.25]; // fixed analog halation colour

function readParams() {
  return {
    halStrength: val("halation_strength"),
    halRadius: val("halation_blur_radius"),
    halThreshold: val("halation_threshold"),
    bloomStrength: val("bloom_strength"),
    bloomRadius: val("bloom_radius"),
    bloomTint: hexToRGB(document.getElementById("bloom_tint_color").value),
    streakStrength: val("streak_strength"),
    streakMode: document.getElementById("streak_mode").value,
    aberration: val("aberration_amount"),
    grain: val("grain_amount") * 0.15,
    fade: val("fade_amount"),
    mute: val("mute_amount"),
    shadows: val("shadows"),
    midtones: val("midtones"),
    highlights: val("highlights"),
    leakStrength: val("leak_strength"),
    leakColor: hexToRGB(document.getElementById("leak_color").value),
    leakPos: val("leak_position"),
  };
}

// ---------------------------------------------------------------
//  Render pipeline
// ---------------------------------------------------------------

function clearTarget(t) {
  bindTarget(t);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

// Threshold-extract + separable blur into result target A (ping-pong with B).
function makeGlow(srcTexture, A, B, threshold, softness, spacing) {
  // bright pass: src (full) -> A (small)
  const pb = progs.bright;
  gl.useProgram(pb);
  useTex(0, srcTexture);
  gl.uniform1i(pb.u("u_src"), 0);
  gl.uniform1f(pb.u("u_threshold"), threshold);
  gl.uniform1f(pb.u("u_softness"), softness);
  bindTarget(A);
  drawQuad();

  // separable gaussian, 2 iterations
  const pblur = progs.blur;
  gl.useProgram(pblur);
  gl.uniform1f(pblur.u("u_spacing"), spacing);
  gl.uniform2f(pblur.u("u_texel"), 1 / A.w, 1 / A.h);
  for (let i = 0; i < 2; i++) {
    // horizontal A -> B
    useTex(0, A.tex);
    gl.uniform1i(pblur.u("u_src"), 0);
    gl.uniform2f(pblur.u("u_dir"), 1, 0);
    bindTarget(B);
    drawQuad();
    // vertical B -> A
    useTex(0, B.tex);
    gl.uniform2f(pblur.u("u_dir"), 0, 1);
    bindTarget(A);
    drawQuad();
  }
}

const STREAK_DIRS = {
  horizontal: [[1, 0]],
  vertical: [[0, 1]],
  cross: [[1, 0], [0, 1]],
  omni: [[1, 0], [0, 1], [0.7071, 0.7071], [0.7071, -0.7071]],
};

function makeStreaks(p) {
  const A = targets.streakA, B = targets.streakB, br = targets.streakBright;
  if (p.streakStrength <= 0) { clearTarget(A); return; }

  // bright extract (high threshold) from base -> streakBright
  const pb = progs.bright;
  gl.useProgram(pb);
  useTex(0, targets.base.tex);
  gl.uniform1i(pb.u("u_src"), 0);
  gl.uniform1f(pb.u("u_threshold"), 0.85);
  gl.uniform1f(pb.u("u_softness"), 4.0);
  bindTarget(br);
  drawQuad();

  // accumulate each direction additively into A
  clearTarget(A);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);
  const ps = progs.streak;
  gl.useProgram(ps);
  gl.uniform2f(ps.u("u_texel"), 1 / A.w, 1 / A.h);
  gl.uniform1f(ps.u("u_spacing"), 6.0);
  const dirs = STREAK_DIRS[p.streakMode] || STREAK_DIRS.horizontal;
  for (const d of dirs) {
    useTex(0, br.tex);
    gl.uniform1i(ps.u("u_src"), 0);
    gl.uniform2f(ps.u("u_dir"), d[0], d[1]);
    bindTarget(A);
    drawQuad();
  }
  gl.disable(gl.BLEND);
}

let needsRender = false;
let downloadEnabled = false;

function render() {
  if (!srcTex || !targets) return;
  const p = readParams();

  // 1. tone curve -> base
  const pt = progs.tone;
  gl.useProgram(pt);
  useTex(0, srcTex);
  gl.uniform1i(pt.u("u_src"), 0);
  gl.uniform1f(pt.u("u_shadows"), p.shadows);
  gl.uniform1f(pt.u("u_midtones"), p.midtones);
  gl.uniform1f(pt.u("u_highlights"), p.highlights);
  bindTarget(targets.base);
  drawQuad();

  // 2. halation glow
  if (p.halStrength > 0) {
    makeGlow(targets.base.tex, targets.halA, targets.halB,
      p.halThreshold, 2.0, p.halRadius * 0.05);
  } else clearTarget(targets.halA);

  // 3. bloom glow
  if (p.bloomStrength > 0) {
    makeGlow(targets.base.tex, targets.bloomA, targets.bloomB,
      0.6, 1.0, p.bloomRadius * 0.05);
  } else clearTarget(targets.bloomA);

  // 4. streaks
  makeStreaks(p);

  // 5. composite -> canvas
  const pc = progs.composite;
  gl.useProgram(pc);
  useTex(0, targets.base.tex);   gl.uniform1i(pc.u("u_base"), 0);
  useTex(1, srcTex);             gl.uniform1i(pc.u("u_src"), 1);
  useTex(2, targets.halA.tex);   gl.uniform1i(pc.u("u_hal"), 2);
  useTex(3, targets.bloomA.tex); gl.uniform1i(pc.u("u_bloom"), 3);
  useTex(4, targets.streakA.tex);gl.uniform1i(pc.u("u_streak"), 4);

  gl.uniform1f(pc.u("u_halStrength"), p.halStrength);
  gl.uniform3fv(pc.u("u_halTint"), HAL_TINT);
  gl.uniform1f(pc.u("u_bloomStrength"), p.bloomStrength);
  gl.uniform3fv(pc.u("u_bloomTint"), p.bloomTint);
  gl.uniform1f(pc.u("u_streakStrength"), p.streakStrength);
  gl.uniform1f(pc.u("u_aberration"), p.aberration);
  gl.uniform1f(pc.u("u_grain"), p.grain);
  gl.uniform1f(pc.u("u_grainColor"), 0.2);
  gl.uniform1f(pc.u("u_mute"), p.mute);
  gl.uniform1f(pc.u("u_fade"), p.fade);
  gl.uniform1f(pc.u("u_seed"), 17.0);

  gl.uniform1f(pc.u("u_leakStrength"), p.leakStrength);
  gl.uniform3fv(pc.u("u_leakColor"), p.leakColor);
  // Position orbits the frame as the slider moves.
  const ang = p.leakPos * Math.PI * 2.0;
  gl.uniform2f(pc.u("u_leakPos"), 0.5 + 0.85 * Math.cos(ang), 0.5 + 0.85 * Math.sin(ang));

  bindTarget(null);
  drawQuad();
}

function requestRender() {
  if (needsRender) return;
  needsRender = true;
  requestAnimationFrame(() => {
    needsRender = false;
    render();
  });
}

// ---------------------------------------------------------------
//  UI wiring
// ---------------------------------------------------------------

function showCanvas() {
  document.getElementById("image-container").classList.remove("empty-state");
  document.getElementById("placeholder").style.display = "none";
  canvas.style.display = "block";
}

document.getElementById("image-upload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    const [pw, ph] = fitTo(
      Math.min(img.naturalWidth, MAX_FULL),
      Math.min(img.naturalHeight, MAX_FULL),
      MAX_PREVIEW
    );
    uploadImage(img);
    allocTargets(pw, ph);
    showCanvas();
    downloadEnabled = true;
    document.getElementById("download-btn").disabled = false;
    render();
    URL.revokeObjectURL(img.src);
  };
  img.onerror = () => fatal("Could not load that image.");
  img.src = URL.createObjectURL(file);
});

// Live updates: any control change re-renders on the next frame.
document.querySelectorAll(
  'input[type="range"], input[type="color"], select'
).forEach((el) => {
  el.addEventListener("input", () => {
    const span = document.getElementById("val-" + el.id);
    if (span) span.textContent = el.value;
    requestRender();
  });
  el.addEventListener("change", requestRender);
});

// Reset all controls to defaults.
document.getElementById("reset-btn").addEventListener("click", () => {
  document.querySelectorAll("input[type='range']").forEach((el) => {
    el.value = el.dataset.default ?? el.value;
    const span = document.getElementById("val-" + el.id);
    if (span) span.textContent = el.value;
  });
  document.getElementById("bloom_tint_color").value = "#ffd799";
  document.getElementById("leak_color").value = "#ff6a2a";
  document.getElementById("streak_mode").value = "horizontal";
  requestRender();
});

// Download — render once at full resolution, then restore preview size.
document.getElementById("download-btn").addEventListener("click", () => {
  if (!downloadEnabled) return;
  const prevW = canvas.width, prevH = canvas.height;
  allocTargets(imgW, imgH);
  render();
  gl.finish();
  canvas.toBlob((blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "halation_result.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    // restore live preview resolution
    allocTargets(prevW, prevH);
    render();
  }, "image/png");
});

// ---------------------------------------------------------------
//  Boot
// ---------------------------------------------------------------
try {
  buildPrograms();
  buildQuad();
} catch (e) {
  fatal("Failed to initialise WebGL: " + e.message);
}