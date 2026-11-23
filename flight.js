// Alps terrain flight: free-flight / walk viewer over the Copernicus GLO-30 tile pyramid.
// Chunks of 225 x 225 cells, quadtree LOD (L6 = 4 deg ... L0 = 1 arcsec), heights decoded from
// Terrarium PNGs into float textures, displaced in the vertex shader. Skirts hide LOD cracks.
import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

// phone build (build_mobile.py sets window.MOBILE): plain map only, 30 m (no fine level), WebP terrain, no satellite /
// painted / sun overlay / 3-D towns, touch controls, everything from the app package (offline)
const MOBILE = !!window.MOBILE;
const TILE_EXT = MOBILE ? 'webp' : 'png';
// quality settings (X / the quality button; remembered per browser). dist: view distance km (0 = unlimited), mesh:
// terrain grid stride (1 = every 30 m sample, 2, 4; lighting keeps the full-resolution normals), res: pixel ratio or
// 'auto' (lowered / raised to hold ~30 fps), detail: LOD split index, logDepth: logarithmic depth buffer (exact far
// away but disables early-z on phone GPUs), aa: antialiasing; these two reload the page when changed
const QS_DEF = MOBILE ? { dist: 25, mesh: 2, res: 'auto', detail: 0, logDepth: 0, labelsMax: 40, aa: 1 }
                      : { dist: 0, mesh: 1, res: 1.5, detail: 1, logDepth: 1, labelsMax: 120, aa: 1 };
const QS = { ...QS_DEF };
try { Object.assign(QS, JSON.parse(localStorage.getItem('flightQuality') || '{}')); } catch (e) { /* defaults */ }
const saveQS = () => { try { localStorage.setItem('flightQuality', JSON.stringify(QS)); } catch (e) { /* */ } };

const meta = await (await fetch('tiles/meta.json')).json();
const DEG = 6371000 * Math.PI / 180;                       // m per degree of latitude
const LAT0 = meta.lat0, LON0 = meta.lon0;
const KX = DEG * Math.cos(LAT0 * Math.PI / 180);            // m per degree of longitude (equirectangular)
const C = meta.chunk, NS = C + 1, PPD = meta.ppd, NL = meta.levels.length;
// optional finer level L = -1 (0.5" ~ 15 m, national DEMs over the Alpine arc; build_fine.py): sparse chunk index
let FINE = null;
if (!MOBILE) try { const r = await fetch('tiles/fine.json'); if (r.ok) FINE = (await r.json()).chunks; } catch (e) { FINE = null; }
const LMIN = FINE ? -1 : 0;

// sample (r, c) of level L -> lon/lat (see build_tiles.py)
const lonOf = (c, L) => meta.west + ((c + 0.5) * 2 ** L - 0.5) / PPD;
const latOf = (r, L) => meta.north - ((r + 0.5) * 2 ** L - 0.5) / PPD;
const xOfLon = lon => (lon - LON0) * KX, zOfLat = lat => -(lat - LAT0) * DEG;   // north = -z
const lonOfX = x => LON0 + x / KX, latOfZ = z => LAT0 - z / DEG;

const PLACES = [  // name, lat, lon, elevation, bearing from the target to the camera (deg, 0 = from north)
  ['Matterhorn', 45.976, 7.658, 4478, 45], ['Monte Rosa', 45.937, 7.867, 4634, 0],
  ['Mont Blanc', 45.833, 6.865, 4808, 330], ['Chamonix', 45.924, 6.870, 1035, 45],
  ['Eiger north face', 46.577, 8.005, 3967, 0], ['Jungfrau', 46.537, 7.963, 4158, 330],
  ['Zermatt', 46.020, 7.749, 1608, 20], ['Aletsch glacier', 46.450, 8.050, 2500, 160],
  ['St. Moritz', 46.498, 9.838, 1822, 330], ['Bernina', 46.382, 9.908, 4049, 0],
  ['Gotthard pass', 46.559, 8.561, 2106, 180], ['Grossglockner', 47.074, 12.694, 3798, 120],
  ['Zugspitze', 47.421, 10.985, 2962, 0], ['Innsbruck', 47.269, 11.404, 574, 0],
  ['Ortler', 46.509, 10.545, 3905, 60], ['Marmolada', 46.434, 11.851, 3343, 0],
  ['Tre Cime', 46.619, 12.302, 2999, 0], ['Cortina', 46.540, 12.136, 1224, 180],
  ['Bolzano', 46.498, 11.354, 262, 200], ['Lake Garda', 45.650, 10.700, 65, 180],
  ['Lake Como', 46.000, 9.250, 199, 180], ['Lugano', 46.004, 8.951, 273, 180],
  ['Aosta', 45.737, 7.320, 583, 160], ['Gran Paradiso', 45.517, 7.267, 4061, 180],
  ['Monviso', 44.667, 7.090, 3841, 90], ['Ecrins', 44.922, 6.360, 4102, 200],
  ['Grenoble', 45.188, 5.724, 214, 270], ['Triglav', 46.378, 13.837, 2864, 180],
  ['Dachstein', 47.475, 13.606, 2995, 0],
];

// ---------------------------------------------------------------- renderer, sky
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: !!QS.aa, logarithmicDepthBuffer: !!QS.logDepth });
const PR_MAX = Math.min(devicePixelRatio, 1.5);
let pixelRatio = QS.res === 'auto' ? Math.min(PR_MAX, 1.0) : Math.min(PR_MAX, QS.res);
renderer.setPixelRatio(pixelRatio);
renderer.setSize(innerWidth, innerHeight, false);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 1, 900000);
camera.rotation.order = 'YXZ';
addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  tunMats.forEach(m => m.resolution.set(innerWidth, innerHeight));
});

const HAZE = [1 / 120000, 1 / 45000, 1 / 400000];
let hazeIdx = 0;
const shared = {
  sunDir: { value: new THREE.Vector3() },
  hazeColor: { value: new THREE.Color(0.73, 0.81, 0.90) },
  hazeDensity: { value: HAZE[0] },
  viewDist: { value: 0 },        // m, 0 = unlimited: terrain fades into the haze over the last 45 %
  sunMode: { value: 0 },
  day: { value: 1 },            // 0 night .. 1 day (from the sun elevation)
  satGlobal: { value: 1 },      // 1 while the map style is satellite (imagery is loaded only then)
  style: { value: 1 },          // T: 0 plain, 1 satellite, 2 painted (land cover computed from the heights)
  doy: { value: 172 },          // day of the year of the simulated clock (seasonal snow line, vegetation)
  on3d: { value: 0 },           // 1 while 3-D towns are shown (painted roofs give way to them nearby)
  waterMask: { value: 0 },      // 1 when water.json (OSM water areas) is loaded: flat ground is water only inside them
  trails: { value: 1 },         // Shift+P: paths & roads overlay on/off (menu: P)
};
const STYLES = ['plain', 'satellite', 'painted'];
// ---------------------------------------------------------------- real sun: NOAA solar calculator (same as
// scripts/terrain_sun.py: ~0.01 deg, Saemundsson refraction) for the simulated clock at the camera position
function sunPosition(tMs, lat, lon) {
  const rad = Math.PI / 180, deg = 180 / Math.PI;
  const jd = tMs / 86400000 + 2440587.5, jc = (jd - 2451545) / 36525;
  const L = ((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360 + 360) % 360;
  const M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
  const ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc);
  const Cc = Math.sin(rad * M) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + Math.sin(2 * rad * M) * (0.019993 - 0.000101 * jc)
    + Math.sin(3 * rad * M) * 0.000289;
  const appLong = L + Cc - 0.00569 - 0.00478 * Math.sin(rad * (125.04 - 1934.136 * jc));
  const obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    + 0.00256 * Math.cos(rad * (125.04 - 1934.136 * jc));
  const decl = Math.asin(Math.sin(rad * obliq) * Math.sin(rad * appLong));
  const y = Math.tan(rad * obliq / 2) ** 2;
  const eqt = 4 * deg * (y * Math.sin(2 * rad * L) - 2 * ecc * Math.sin(rad * M) + 4 * ecc * y * Math.sin(rad * M) * Math.cos(2 * rad * L)
    - 0.5 * y * y * Math.sin(4 * rad * L) - 1.25 * ecc * ecc * Math.sin(2 * rad * M));
  const minutes = ((tMs / 60000) % 1440 + 1440) % 1440;
  const tst = ((minutes + eqt + 4 * lon) % 1440 + 1440) % 1440;
  const ha = rad * (tst / 4 - 180), phi = rad * lat;
  const cosz = Math.max(-1, Math.min(1, Math.sin(phi) * Math.sin(decl) + Math.cos(phi) * Math.cos(decl) * Math.cos(ha)));
  const el = 90 - deg * Math.acos(cosz);
  const az = (deg * Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(phi) - Math.tan(decl) * Math.cos(phi)) + 180 + 360) % 360;
  const refr = el > -1 ? 1.02 / Math.tan(rad * (el + 10.3 / (el + 5.11))) / 60 : 0;
  return [az, el + refr];
}
let simTime = Date.now();       // simulated clock (ms UTC); runs in real time, [ ] , . K shift it
{ const t = Date.parse(new URLSearchParams(location.search).get('t') || ''); if (Number.isFinite(t)) simTime = t; }  // ?t=2026-07-15T12:00
let sunAz = 150, sunEl = 35;    // degrees, azimuth clockwise from north (updated every frame)
function setSun(lat, lon) {
  [sunAz, sunEl] = sunPosition(simTime, lat, lon);
  // light direction: below the horizon keep a dim light from just above it (twilight), fade with shared.day
  const a = sunAz * Math.PI / 180, e = Math.max(sunEl, 1) * Math.PI / 180;
  shared.sunDir.value.set(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e));
  shared.day.value = Math.min(1, Math.max(0, (sunEl + 6) / 12));          // civil twilight ramp -6..+6 deg
  const d = new Date(simTime);
  shared.doy.value = (simTime - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400e3;
}
const fmtTime = t => new Date(t).toLocaleString('en-GB', { timeZone: 'Europe/Zurich', weekday: 'short', day: '2-digit',
  month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });

const sky = new THREE.Mesh(new THREE.SphereGeometry(1000, 32, 16), new THREE.ShaderMaterial({
  uniforms: { sunDir: shared.sunDir, hazeColor: shared.hazeColor, day: shared.day },
  vertexShader: `varying vec3 vDir;
    void main() { vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform vec3 sunDir; uniform vec3 hazeColor; uniform float day; varying vec3 vDir;
    void main() {
      vec3 d = normalize(vDir);
      vec3 zenith = vec3(0.25, 0.46, 0.80);
      vec3 c = d.y > 0.0 ? mix(hazeColor, zenith, pow(d.y, 0.5)) : hazeColor * (1.0 + 0.3 * d.y);
      float s = max(dot(d, sunDir), 0.0);
      float low = 1.0 - smoothstep(0.0, 0.25, sunDir.y);                       // warm glow near the horizon
      c = mix(c, vec3(0.95, 0.55, 0.32), low * pow(s, 6.0) * 0.6 * day);
      c *= mix(0.04, 1.0, day);
      c += vec3(1.0, 0.92, 0.75) * (pow(s, 900.0) * 3.0 + pow(s, 16.0) * 0.18) * smoothstep(0.3, 0.7, day);
      gl_FragColor = vec4(c, 1.0);
    }`,
  side: THREE.BackSide, depthWrite: false, depthTest: false,
}));
sky.renderOrder = -1; sky.frustumCulled = false;
scene.add(sky);

// ---------------------------------------------------------------- terrain shaders, shared grid
const VERT = `
#include <common>
#include <logdepthbuf_pars_vertex>
uniform sampler2D hmap; uniform vec2 origin; uniform vec2 stepv; uniform float skirtDepth;
uniform vec4 sunXf; uniform vec4 satXf;
attribute vec2 grid; attribute float skirt;
varying vec3 vWorld; varying vec3 vNormal; varying float vFlat; varying vec2 vSunUV; varying vec2 vSatUV;
varying float vCurv; varying vec2 vFeatUV; varying float vSea; varying float vBuilt;
float H(ivec2 p) { p = clamp(p, ivec2(0), ivec2(${C})); return texelFetch(hmap, p, 0).r; }
void main() {
  ivec2 p = ivec2(grid + 0.5);
  float h = H(p);
  ivec2 a = clamp(p - 1, ivec2(0), ivec2(${C})), b = clamp(p + 1, ivec2(0), ivec2(${C}));
  float dhx = (H(ivec2(b.x, p.y)) - H(ivec2(a.x, p.y))) / (float(b.x - a.x) * stepv.x);
  float dhz = (H(ivec2(p.x, b.y)) - H(ivec2(p.x, a.y))) / (float(b.y - a.y) * stepv.y);
  vNormal = normalize(vec3(-dhx, 1.0, -dhz));
  vFlat = (abs(dhx) + abs(dhz) < 1e-4) ? 1.0 : 0.0;           // lakes are exactly flat in GLO-30 ...
  if (vFlat > 0.5) {                                           // ... over a wider patch too (the phone tiles round to
    float f2 = abs(H(p + ivec2(2, 0)) - h) + abs(H(p - ivec2(2, 0)) - h)          // 1 m, so small level patches of
             + abs(H(p + ivec2(0, 2)) - h) + abs(H(p - ivec2(0, 2)) - h);         // a plain are not lakes)
    vFlat = f2 < 1e-3 ? 1.0 : 0.0;
  }
  vSea = abs(h) < 0.002 ? 1.0 : 0.0;                           // GLO-30 sea and lagoons are exactly 0 m; land below
                                                               // sea level (reclaimed plains) is not
  vBuilt = texelFetch(hmap, p, 0).g;                           // phone tiles: GHSL built-up share (0 on the PC)
  // concavity: mean of the 4 neighbours 2 samples away minus h, per metre (> 0 gully/valley, < 0 ridge)
#ifdef LITE
  vCurv = 0.0;
#else
  float hn = 0.25 * (H(p + ivec2(2, 0)) + H(p - ivec2(2, 0)) + H(p + ivec2(0, 2)) + H(p - ivec2(0, 2)));
  vCurv = (hn - h) / (2.0 * stepv.y);
#endif
  vWorld = vec3(origin.x + grid.x * stepv.x, h - skirt * skirtDepth, origin.y + grid.y * stepv.y);
  vSunUV = sunXf.xy + grid * sunXf.zw;
  vSatUV = satXf.xy + grid * satXf.zw;
  vFeatUV = grid / ${C}.0;                                       // feature mask covers the chunk exactly
  gl_Position = projectionMatrix * viewMatrix * vec4(vWorld, 1.0);
  #include <logdepthbuf_vertex>
}`;
const FRAG = `
#include <common>
#include <logdepthbuf_pars_fragment>
uniform vec3 sunDir; uniform vec3 hazeColor; uniform float hazeDensity; uniform float viewDist;
uniform sampler2D sunTex; uniform float sunMode; uniform float day;
uniform sampler2D satTex; uniform float satOn; uniform float satGlobal; uniform float style; uniform float doy;
uniform sampler2D featTex; uniform float featOn; uniform float on3d;
uniform sampler2D trailTex; uniform float trailOn; uniform float trails; uniform float waterMask; uniform float featValid;
varying vec3 vWorld; varying vec3 vNormal; varying float vFlat; varying vec2 vSunUV; varying vec2 vSatUV;
varying float vCurv; varying vec2 vFeatUV; varying float vSea; varying float vBuilt;
vec3 magma(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  vec3 c0 = vec3(0.03, 0.02, 0.10), c1 = vec3(0.32, 0.07, 0.43), c2 = vec3(0.72, 0.21, 0.47),
       c3 = vec3(0.98, 0.47, 0.36), c4 = vec3(0.99, 0.95, 0.72);
  return t < 1.0 ? mix(c0, c1, t) : t < 2.0 ? mix(c1, c2, t - 1.0) : t < 3.0 ? mix(c2, c3, t - 2.0) : mix(c3, c4, t - 3.0);
}
// sin-free hash: world coordinates reach ~6e5 m, where sin() of large arguments bands on many GPUs
float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float rnoise(vec2 p) { return 1.0 - abs(2.0 * vnoise(p) - 1.0) * 0.7 - abs(2.0 * vnoise(p * 2.1 + 5.3) - 1.0) * 0.3; }
float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s / 0.9375; }
vec3 lin(vec3 c) { return pow(c, vec3(2.2)); }
// cell noise: distance to the nearest jittered point (cell units), vector to it and an id per cell
float worley(vec2 x, out vec2 toC, out float id) {
  vec2 i = floor(x), f = fract(x); float d = 9.0; toC = vec2(0.0); id = 0.0;
  for (int jy = -1; jy <= 1; jy++) for (int jx = -1; jx <= 1; jx++) {
    vec2 g = vec2(float(jx), float(jy)), o = vec2(hash(i + g), hash(i + g + 19.7));
    vec2 r = g + o - f; float dd = dot(r, r);
    if (dd < d) { d = dd; toC = r; id = hash(i + g + 3.3); }
  }
  return sqrt(d);
}
// piecewise-linear seasonal snow line [m] over the day of the year (northern Alps climatology, rough)
float snowLine(float t) {
  t = mod(t, 365.0);
  return t < 60.0 ? mix(800.0, 950.0, t / 60.0) : t < 125.0 ? mix(950.0, 1800.0, (t - 60.0) / 65.0)
       : t < 175.0 ? mix(1800.0, 2600.0, (t - 125.0) / 50.0) : t < 235.0 ? mix(2600.0, 3050.0, (t - 175.0) / 60.0)
       : t < 275.0 ? mix(3050.0, 2850.0, (t - 235.0) / 40.0) : t < 310.0 ? mix(2850.0, 1900.0, (t - 275.0) / 35.0)
       : t < 340.0 ? mix(1900.0, 1100.0, (t - 310.0) / 30.0) : mix(1100.0, 800.0, (t - 340.0) / 25.0);
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
// grain colour over the farming year (winter grain; spring crops run ~25 days later)
vec3 grainCol(float t) {
  vec3 c = vec3(0.40, 0.37, 0.26);                                   // winter: bare / sprouting
  c = mix(c, vec3(0.40, 0.52, 0.25), smoothstep(85.0, 125.0, t));    // green
  c = mix(c, vec3(0.76, 0.65, 0.34), smoothstep(160.0, 195.0, t));   // ripe
  c = mix(c, vec3(0.70, 0.64, 0.47), smoothstep(205.0, 220.0, t));   // stubble after harvest
  c = mix(c, vec3(0.43, 0.34, 0.25), smoothstep(255.0, 290.0, t));   // ploughed
  return c;
}
// "painted" map, medieval: land cover guessed from height, slope, aspect, concavity, latitude and the date, plus
// settlements / castles / rivers from OpenStreetMap rasterised per chunk (featTex: r river, g town 0.55 / wall 1,
// b cultivation = nearness to settlements). No imagery. nOff: normal offset for roofs.
#ifndef LITE
vec3 painted(float h, vec3 n, float dist, out float snowOut, out float rockOut, out vec3 nOff) {
  vec2 p = vWorld.xz;
  nOff = vec3(0.0);
  float lat = ${LAT0.toFixed(4)} - vWorld.z / ${DEG.toFixed(3)};
  float steep = 1.0 - n.y;                                         // 0 flat .. ~0.5 at 60 deg
  float hz = length(n.xz);
  float north = hz > 1e-3 ? -n.z / hz * smoothstep(0.03, 0.15, hz) : 0.0;   // +1 north-facing, -1 south-facing
  float big = fbm(p / 2500.0), mid = fbm(p / 400.0);
  // fade small patterns to their mean once they shrink below ~2 px (otherwise they alias into speckle)
  float px = length(fwidth(p));                                    // metres per pixel here
  float fine = mix(vnoise(p / 35.0), 0.5, smoothstep(10.0, 30.0, px));
  float curv = clamp(vCurv * 4.0, -1.0, 1.0);                      // + gully / valley floor, - ridge
  vec4 F = featOn > 0.5 ? texture2D(featTex, vFeatUV) : vec4(0.0, 0.0, 0.0, 0.0);
  float cult = featOn > 0.5 ? F.b : 0.45 * smoothstep(0.4, 0.6, mid) * smoothstep(1400.0, 600.0, h);
  // tree line: ~1850 m at the northern and southern rims, ~2250 m in the inner Alps; south slopes higher
  float tl = 1850.0 + 400.0 * exp(-pow((lat - 46.4) / 0.45, 2.0)) - 120.0 * north + (big - 0.5) * 300.0;
  // season: greenness rises in spring (later higher up) and fades in autumn
  float t = doy - h / 1000.0 * 12.0;
  float green = smoothstep(85.0, 140.0, t) * (1.0 - smoothstep(275.0, 320.0, doy + h / 1000.0 * 10.0));
  float autumn = smoothstep(265.0, 290.0, doy) * (1.0 - smoothstep(305.0, 330.0, doy));
  // ---- wild land: forest wherever nobody farms; crowns drawn individually up close (cell noise), bumped in the lighting
  float lon = ${LON0.toFixed(4)} + vWorld.x / ${KX.toFixed(3)};
  float coniferShare = smoothstep(700.0, 1300.0, h + (mid - 0.5) * 400.0);
  vec3 broad = mix(mix(vec3(0.30, 0.27, 0.23), vec3(0.20, 0.33, 0.13), green), vec3(0.58, 0.36, 0.13), autumn * 0.8);
  vec3 conif = vec3(0.10, 0.19, 0.11);
  vec3 forestBase = mix(broad, conif, coniferShare);
  float crownDetail = 1.0 - smoothstep(1.2, 4.0, px);
  vec3 crownN = vec3(0.0);
  vec3 forest = forestBase * (0.82 + 0.3 * fine);
  if (crownDetail > 0.01) {
    vec2 cr; float cid;
    float f1 = worley(p / mix(8.0, 5.5, coniferShare), cr, cid);
    float crown = smoothstep(0.62, 0.25, f1);                        // dome of one crown, dark gaps between
    vec3 fc = forestBase * (0.88 + 0.28 * cid) * mix(0.45, 1.05, crown);
    fc = mix(fc, vec3(0.62, 0.50, 0.18), autumn * coniferShare * step(0.8, cid) * smoothstep(tl - 500.0, tl - 200.0, h));  // larches
    forest = mix(forest * 0.85, fc, crownDetail);
    crownN = vec3(-cr.x, 0.0, -cr.y) * 1.2 * crown * crownDetail;
  }
  forest = mix(forest, vec3(0.62, 0.50, 0.18), autumn * smoothstep(tl - 400.0, tl - 150.0, h) * smoothstep(0.5, 0.65, mid) * (1.0 - crownDetail));
  // ---- tree line: dense forest thins into clumps of trees, then open alpine grassland
  float dens = 1.0 - smoothstep(tl - 260.0, tl + 90.0, h + (mid - 0.5) * 160.0 + (fine - 0.5) * 60.0);
  float trees = dens;
  if (dens > 0.0 && dens < 1.0 && px < 12.0) {
    vec2 kr; float kid;
    float k1 = worley(p / 16.0, kr, kid);
    float clump = smoothstep(0.12, -0.08, k1 - 0.75 * sqrt(dens) + 0.15 * (vnoise(p / 6.0) - 0.5));   // ragged clumps thin out toward the line
    trees = mix(clump, dens, smoothstep(4.0, 12.0, px));
  }
  float above = 1.0 - trees;
  vec3 c = forest;
  // ---- open fields around the villages: strips (~18 m x 120 m) in furlongs of different orientation
  float tilled = smoothstep(0.25, 0.55, cult + 0.25 * (mid - 0.5)) * smoothstep(0.17, 0.08, steep) * smoothstep(1700.0, 1150.0, h)
               * smoothstep(0.72, 0.6, mid);                         // woods left standing between the villages
  float pasture = smoothstep(0.04, 0.25, cult + 0.2 * (mid - 0.5)) * smoothstep(0.38, 0.22, steep) * smoothstep(tl, tl - 250.0, h);
  // lowland hay meadow / pasture: lush, even green with faint mowing swaths
  vec3 meadow = mix(vec3(0.46, 0.46, 0.30), vec3(0.36, 0.55, 0.20), green) * (0.93 + 0.14 * fine);
  meadow *= 1.0 + 0.05 * sin(dot(p, vec2(0.6, 0.8)) / 1.6) * (1.0 - smoothstep(1.0, 3.0, px));
  c = mix(c, meadow, pasture);
  if (tilled > 0.01) {
    vec2 fb = floor(p / 240.0);
    vec2 r = rot(hash(fb) * 3.1416) * p;
    float sx = r.x / 18.0, sy = r.y / 120.0;
    float id = hash(vec2(floor(sx), floor(sy)) + fb * 7.13);
    vec3 strip = id < 0.3 ? grainCol(doy) : id < 0.5 ? grainCol(doy - 25.0) : meadow;     // winter / spring crop / hay + fallow
    strip = mix(meadow, strip, 0.75);                                 // muted: hand-sown crops, weedy, uneven
    strip *= 0.94 + 0.12 * hash(vec2(floor(sx), floor(sy)) + 3.1);
    float balk = smoothstep(0.0, 0.08, fract(sx)) * smoothstep(0.0, 0.02, fract(sy));      // grass balks between strips
    strip = mix(meadow * 0.85, strip, balk);
    vec3 mean = mix(meadow, 0.3 * grainCol(doy) + 0.2 * grainCol(doy - 25.0) + 0.5 * meadow, 0.75);
    strip = mix(strip, mean, smoothstep(6.0, 20.0, px));
    vec2 fe = abs(fract(p / 240.0) - 0.5);                            // hedgerows along furlong block edges
    float hedge = max(smoothstep(0.475, 0.49, fe.x) * step(0.45, hash(fb + 2.0)), smoothstep(0.475, 0.49, fe.y) * step(0.45, hash(fb + 5.0)));
    strip = mix(strip, forestBase * 0.9, hedge * (1.0 - smoothstep(4.0, 12.0, px)));
    c = mix(c, strip, tilled);
  }
  // ---- vineyards on warm south slopes near settlements (south of the main ridge and in the dry inner valleys)
  float vine = smoothstep(-0.25, -0.55, north) * smoothstep(0.04, 0.09, steep) * smoothstep(0.4, 0.3, steep)
             * smoothstep(950.0, 750.0, h) * smoothstep(0.25, 0.45, cult) * smoothstep(46.75, 46.55, lat);
  if (vine > 0.01) {
    float row = fract(h / 2.2);                                       // rows follow the contour lines
    vec3 leaf = mix(mix(vec3(0.42, 0.36, 0.26), vec3(0.30, 0.42, 0.16), green), vec3(0.62, 0.40, 0.14), autumn);
    vec3 vc = mix(vec3(0.58, 0.50, 0.38), leaf, smoothstep(0.35, 0.5, row) * smoothstep(0.95, 0.8, row));
    c = mix(c, mix(vc, mix(leaf, vec3(0.58, 0.50, 0.38), 0.4), smoothstep(1.5, 4.0, px)), vine);
  }
  // ---- alpine grassland above the trees: tawny-olive, tussocky, dwarf-shrub heath near the line, stones higher up
  vec3 alpGreen = vec3(0.60, 0.60, 0.34), alpDry = vec3(0.70, 0.60, 0.40);
  vec3 alpine = mix(alpDry, alpGreen, green * (1.0 - 0.5 * autumn)) * (0.88 + 0.22 * fine);
  alpine *= 0.92 + 0.16 * vnoise(p / 3.0) * (1.0 - smoothstep(1.0, 3.0, px));             // tussocks
  float heath = smoothstep(tl + 380.0, tl + 60.0, h) * smoothstep(0.48, 0.62, vnoise(p / 45.0) * 0.6 + mid * 0.4);
  vec3 heathC = mix(vec3(0.29, 0.32, 0.19), vec3(0.58, 0.24, 0.16), autumn);             // alpenrose / bilberry, red in autumn
  alpine = mix(alpine, heathC * (0.85 + 0.3 * fine), heath);
  float stony = smoothstep(tl + 250.0, tl + 900.0, h);
  if (px < 3.0 && stony > 0.01) {
    vec2 sr; float sid;
    float s1 = worley(p / 3.5, sr, sid);
    alpine = mix(alpine, vec3(0.64, 0.62, 0.57) * (0.85 + 0.25 * sid), smoothstep(0.32, 0.22, s1 + 0.25 * (1.0 - stony)) * (1.0 - smoothstep(1.5, 3.0, px)));
  }
  alpine = mix(alpine, vec3(0.58, 0.56, 0.48), 0.6 * stony * smoothstep(3.0, 8.0, px));
  c = mix(c, alpine, above * (1.0 - tilled));
  // ---- rock: lithology by region (northern / southern limestone Alps and Dolomites pale, crystalline core darker),
  // tilted strata, joints and fractures, dark water streaks on steep faces, lichen on gentler slabs
  float soilLoss = smoothstep(tl + 200.0, tl + 900.0, h);
  float rock = smoothstep(0.30 - 0.20 * soilLoss, 0.42 - 0.22 * soilLoss, steep + 0.10 * (mid - 0.5) - 0.06 * curv);
  rock = max(rock, above * smoothstep(0.62, 0.75, mid + 0.25 * soilLoss) * smoothstep(0.05, 0.15, steep));   // outcrops
  float dolo = smoothstep(46.1, 46.25, lat) * smoothstep(46.75, 46.6, lat) * smoothstep(11.35, 11.55, lon) * smoothstep(12.55, 12.35, lon);
  float limeReg = max(max(smoothstep(47.2, 47.4, lat), smoothstep(46.0, 45.85, lat)), dolo);
  float lime = clamp(limeReg + (big - 0.5) * 0.7, 0.0, 1.0);
  vec3 crystal = mix(vec3(0.50, 0.48, 0.45), vec3(0.45, 0.48, 0.44), smoothstep(0.4, 0.6, mid));   // gneiss / schist
  vec3 limeC = mix(vec3(0.76, 0.74, 0.69), vec3(0.84, 0.76, 0.67), dolo);                         // limestone / dolomite
  vec3 rockCol = mix(crystal, limeC, smoothstep(0.35, 0.65, lime));
  float rd = 1.0 - smoothstep(6.0, 25.0, px);                                                       // rock detail
  vec2 dip = 0.25 * vec2(cos(big * 9.0), sin(big * 9.0));
  float band = (h + dot(p, dip)) / mix(2.5, 7.0, lime);
  float strata = vnoise(vec2(band, dot(p, dip.yx) / 300.0));
  rockCol *= mix(1.0, 0.72 + 0.5 * strata, mix(0.4, 1.0, lime) * rd);
  rockCol *= 1.0 - 0.18 * smoothstep(0.1, 0.0, fract(band)) * lime * rd;                            // bedding planes
  rockCol *= 0.85 + 0.3 * vnoise(p / 90.0);                                                          // big tonal patches
  vec2 fd = hz > 1e-3 ? n.xz / hz : vec2(0.0, 1.0);                                                 // downslope
  vec2 pp = vec2(-fd.y, fd.x);
  float joints = 1.0 - abs(2.0 * vnoise(p / 26.0 + 3.0 * vec2(vnoise(p / 70.0), vnoise(p / 70.0 + 7.0))) - 1.0);
  float jdens = smoothstep(0.35, 0.75, vnoise(p / 220.0 + 11.0));                                   // fractured zones vary
  rockCol *= 1.0 - 0.5 * smoothstep(0.955, 0.99, joints) * jdens * rd;                               // fractures
  float streak = vnoise(vec2(dot(p, pp) / 2.2, h / 45.0));
  rockCol *= 1.0 - 0.35 * smoothstep(0.55, 0.85, streak) * smoothstep(0.3, 0.5, steep) * rd;         // water stains
  float lich = smoothstep(0.62, 0.75, vnoise(p / 5.0)) * smoothstep(0.45, 0.25, steep) * rd;
  rockCol = mix(rockCol, mix(vec3(0.42, 0.45, 0.36), vec3(0.70, 0.52, 0.25), step(0.85, hash(floor(p / 5.0)))), 0.5 * lich);
  vec3 scree = mix(crystal, limeC, smoothstep(0.35, 0.65, lime)) * (0.92 + 0.2 * fine);
  if (px < 3.0) { vec2 gr; float gid; float g1 = worley(p / 1.6, gr, gid); scree *= mix(1.0, 0.75 + 0.4 * gid * smoothstep(0.1, 0.5, g1), 1.0 - smoothstep(1.0, 3.0, px)); }
  float screeW = soilLoss * smoothstep(0.08, 0.16, steep) * (1.0 - rock) * smoothstep(-0.1, 0.25, curv);
  c = mix(c, scree, max(screeW, 0.7 * soilLoss * smoothstep(0.55, 0.75, fine + 0.3 * curv)));
  c = mix(c, rockCol, rock);
  rockOut = rock;
  nOff = crownN * trees * (1.0 - tilled) * (1.0 - pasture) * (1.0 - rock);
  // ---- snow: seasonal snow line (lower on north slopes), steep faces shed it, forest canopy hides some
  float sl = snowLine(doy) - 220.0 * north + (big - 0.5) * 250.0;
  float s = smoothstep(sl - 120.0, sl + 160.0, h + (fine - 0.5) * 120.0 + 80.0 * curv);
  s *= mix(1.0, 0.25, smoothstep(0.45, 0.7, steep));
  float wild = trees * (1.0 - max(tilled, pasture)) * (1.0 - rock);
  s *= mix(1.0, 0.45, wild);                                                       // canopy hides part of it
  float ice = smoothstep(2850.0, 3150.0, h - 150.0 * north + 120.0 * curv + (big - 0.5) * 300.0) * smoothstep(0.42, 0.25, steep);
  vec3 snowCol = vec3(0.95, 0.96, 0.99);
  c = mix(c, mix(snowCol, vec3(0.80, 0.88, 0.95), 0.35 * (1.0 - s)), max(s, ice));
  snowOut = max(s, ice);
  c *= 1.0 - 0.25 * max(curv, 0.0) + 0.08 * max(-curv, 0.0);
  // ---- rivers: open water with gravel banks (unregulated, braided)
  float rv = F.r;                                                  // water coverage
  float grav = smoothstep(0.06, 0.2, F.g) * (1.0 - smoothstep(0.27, 0.33, F.g));   // gravel bank (g 60)
  c = mix(c, vec3(0.60, 0.57, 0.50) * (0.9 + 0.2 * fine), grav * (1.0 - s * 0.8));
  c = mix(c, vec3(0.17, 0.30, 0.35), smoothstep(0.05, 0.6, rv));
  // ---- towns: houses in blocks along muddy lanes, pitched roofs, stone walls
  float town = smoothstep(0.30, 0.45, F.g) * (1.0 - smoothstep(0.6, 0.75, rv));
  float wall = smoothstep(0.80, 0.93, F.g);
  if (town > 0.01) {
    float south = smoothstep(46.45, 46.15, lat + 0.3 * (big - 0.5));      // terracotta south of the main ridge
    vec2 r = rot(vnoise(p / 700.0) * 3.1416) * p + 16.0 * vec2(vnoise(p / 85.0), vnoise(p / 85.0 + 9.0)) + 6.0 * vec2(vnoise(p / 23.0), vnoise(p / 23.0 + 4.0));
    vec2 br = r / vec2(30.0, 22.0), bf = fract(br), bi = floor(br);
    float lane = 1.0 - smoothstep(0.10, 0.14, bf.x) * smoothstep(0.13, 0.18, bf.y);
    vec2 hq = (bf - vec2(0.14, 0.18)) / vec2(0.86, 0.82) * vec2(3.0, 2.0);
    vec2 hc = floor(hq), hf = fract(hq);
    float id = hash(bi * 5.0 + hc);
    vec3 roofS = id < 0.4 ? vec3(0.66, 0.33, 0.22) : id < 0.75 ? vec3(0.58, 0.30, 0.21) : id < 0.92 ? vec3(0.72, 0.42, 0.28) : vec3(0.52, 0.48, 0.42);
    vec3 roofN = id < 0.35 ? vec3(0.60, 0.32, 0.22) : id < 0.65 ? vec3(0.50, 0.47, 0.44) : id < 0.85 ? vec3(0.42, 0.33, 0.26) : vec3(0.36, 0.37, 0.40);
    vec3 roof = mix(roofN, roofS, south) * (0.9 + 0.2 * hash(bi + hc * 3.0 + 1.7));
    float yard = step(0.86, hash(bi * 3.0 + hc + 9.1));                    // some plots are gardens / yards
    float eave = smoothstep(0.0, 0.06, hf.x) * smoothstep(1.0, 0.94, hf.x) * smoothstep(0.0, 0.05, hf.y) * smoothstep(1.0, 0.95, hf.y);
    vec3 mud = vec3(0.58, 0.53, 0.44);
    vec3 hcol = mix(mud * 0.85, mix(roof, meadow * 0.9, yard), eave);
    vec3 tc = mix(hcol, mud, lane);
    vec3 tmean = mix(mix(mix(vec3(0.50, 0.38, 0.30), vec3(0.63, 0.33, 0.23), south), mud, 0.35), meadow, 0.06);
    float detail = 1.0 - smoothstep(3.0, 9.0, px);
    tc = mix(tmean, tc, detail);
    // within the 3-D range the buildings stand on their own: ground between them is lanes and yards
    float near3 = on3d * (1.0 - smoothstep(5000.0, 6000.0, dist));
    tc = mix(tc, mix(mud, meadow * 0.9, 0.35 + 0.3 * fine), near3);
    tc = mix(tc, snowCol * 0.97, s * (0.85 - 0.3 * lane));                 // snow on roofs in winter
    c = mix(c, tc, town);
    // roofs: ridge along the house's long (y) side, the two halves face opposite ways
    float pitch = (1.0 - lane) * (1.0 - yard) * eave * (1.0 - smoothstep(1.5, 3.0, px)) * town * (1.0 - near3);
    vec2 dir = (hf.x < 0.5 ? -1.0 : 1.0) * vec2(1.0, 0.0) * rot(vnoise(p / 700.0) * 3.1416);
    nOff = nOff * (1.0 - town) + vec3(dir.x, 0.0, dir.y) * 0.6 * pitch;
  }
  wall *= 1.0 - on3d * (1.0 - smoothstep(5000.0, 6000.0, dist));
  if (wall > 0.01) {
    vec3 stone = vec3(0.70, 0.66, 0.57) * (0.9 + 0.15 * fine);
    c = mix(c, mix(stone, snowCol, s * 0.5), wall);
  }
  return c;
}
#endif
void main() {
  #include <logdepthbuf_fragment>
  vec3 n = normalize(vNormal);
  float h = vWorld.y;
  float dist = length(vWorld - cameraPosition);
  bool sat = satOn > 0.5 && satGlobal > 0.5;
  bool paint = style > 1.5;
  vec3 c;
  float snowP = 0.0, rockP = 0.0, riv = 0.0;
#ifndef LITE
  if (paint) {
    vec3 nOff;
    c = painted(h, n, dist, snowP, rockP, nOff);
    n = normalize(n + nOff);
    // procedural micro-relief in the lighting: rock faces get more, fades out with distance
    float amp = (0.10 + 0.15 * rockP) * (1.0 - snowP * 0.6) * (1.0 - smoothstep(4.0, 12.0, length(fwidth(vWorld.xz))));
    if (amp > 0.002 && vFlat < 0.5) {
      vec2 p = vWorld.xz / 28.0; float e = 0.35;
      float b0 = fbm(p), bx = fbm(p + vec2(e, 0.0)), bz = fbm(p + vec2(0.0, e));
      n = normalize(n - amp * vec3(bx - b0, 0.0, bz - b0) / e * 2.0);
      if (rockP > 0.05) {                    // rock: sharp ridged relief (ledges, ribs) at 9 m and 2.5 m
        vec2 q = vWorld.xz / 9.0; float e2 = 0.12;
        float r0 = rnoise(q), rx = rnoise(q + vec2(e2, 0.0)), rz = rnoise(q + vec2(0.0, e2));
        vec2 q2 = vWorld.xz / 2.5;
        float s0 = rnoise(q2), sx = rnoise(q2 + vec2(e2, 0.0)), sz = rnoise(q2 + vec2(0.0, e2));
        float ra = rockP * (1.0 - snowP * 0.7) * (1.0 - smoothstep(3.0, 10.0, length(fwidth(vWorld.xz))));
        n = normalize(n - ra * (0.35 * vec3(rx - r0, 0.0, rz - r0) + 0.18 * vec3(sx - s0, 0.0, sz - s0)) / e2);
      }
    }
  } else
#endif
  {
    float nz = 0.6 * vnoise(vWorld.xz / 450.0) + 0.4 * vnoise(vWorld.xz / 70.0);
    float hh = h + (nz - 0.5) * 300.0;
    c = mix(vec3(0.43, 0.53, 0.29), vec3(0.20, 0.31, 0.17), smoothstep(450.0, 900.0, hh));  // fields -> forest
    c = mix(c, vec3(0.52, 0.57, 0.34), smoothstep(1750.0, 2050.0, hh));                         // alpine meadow
    c = mix(c, vec3(0.60, 0.58, 0.54), smoothstep(2350.0, 2800.0, hh));                         // scree
    c = mix(c, vec3(0.47, 0.44, 0.41), smoothstep(0.86, 0.70, n.y));                            // steep = rock
    c *= 0.88 + 0.24 * nz;
    // permanent ice above ~3000 m; seasonal snow down to the snow line of the date (~800 m in mid-winter)
    float sline = min(3000.0, snowLine(doy));
    float snow = max(smoothstep(2950.0, 3300.0, hh), smoothstep(sline - 100.0, sline + 250.0, hh))
               * smoothstep(0.62, 0.82, n.y);
    c = mix(c, vec3(0.96, 0.97, 1.0), snow);
    // rivers from the OSM mask (same as the painted map): gravel banks, then water in the lake colour
    if (featOn > 0.5) {
      vec4 F = texture2D(featTex, vFeatUV);
      riv = F.r;
      float grav = smoothstep(0.06, 0.2, F.g) * (1.0 - smoothstep(0.27, 0.33, F.g));
      c = mix(c, vec3(0.60, 0.57, 0.50) * (0.92 + 0.16 * nz), grav * (1.0 - snow * 0.8));
      c = mix(c, vec3(0.14, 0.29, 0.42), smoothstep(0.05, 0.6, riv));
    }
  }
  if (!paint && vBuilt > 0.02) {   // towns and villages (GHSL built-up share): beige roofs, grey city cores
    vec3 town = mix(vec3(0.74, 0.66, 0.58), vec3(0.64, 0.62, 0.62), smoothstep(0.3, 0.7, vBuilt));
    c = mix(c, town, 0.9 * smoothstep(0.04, 0.45, vBuilt));
  }
  if (sat) c = texture2D(satTex, vSatUV).rgb * 1.08;          // Sentinel-2 cloudless (sRGB)
  // water: sea level; OSM water areas + rivers from the feature mask (plain, painted); exactly flat ground only where
  // there is no mask (far chunks, satellite glint, water.json missing): GLO-30 is also flat on wetlands and plains
  if (paint && featOn > 0.5) riv = texture2D(featTex, vFeatUV).r;
  bool masked = featOn > 0.5 && !sat && waterMask > 0.5 && featValid > 0.5;   // outside the OSM extract: flat test
#ifdef LITE
  bool water = vSea > 0.5 || (riv > 0.7 && !sat);               // phone: sea + OSM water only (1-m heights: plains
#else                                                           // look exactly flat, so no flatness guess)
  bool water = vSea > 0.5 || (vFlat > 0.5 && !masked) || (riv > 0.7 && !sat);
#endif
  float land = 0.0;
  if (sunMode > 0.5) {                     // sun-hours / sunless-days overlay (scripts/sun_alps.py)
    vec4 sv = texture2D(sunTex, vSunUV);
    land = sv.b;
    if (sunMode < 1.5) c = mix(c, magma(sv.r * 255.0 / 254.0), 0.92 * land);
    else {
      float days = sv.g * 255.0;
      vec3 oc = mix(vec3(0.62, 0.78, 0.92), vec3(0.03, 0.15, 0.40), clamp(days / 150.0, 0.0, 1.0));
      c = mix(mix(c, vec3(0.85), 0.35 * land), oc, 0.92 * land * smoothstep(0.0, 25.0, days));   // soft edge
    }
  }
  if (water && paint && snowP > 0.6 && h > 1300.0) water = false;          // frozen, snow-covered mountain lake
  if (water && land < 0.5 && !sat) c = paint ? vec3(0.10, 0.25, 0.36) : vec3(0.14, 0.29, 0.42);
  if (trailOn > 0.5 && trails > 0.5) {     // paths & roads overlay (build_trails.py), premultiplied colour
    vec4 T = texture2D(trailTex, vFeatUV);
    float fade = 1.0 - smoothstep(9000.0, 20000.0, dist);
    c = c * (1.0 - T.a * fade) + T.rgb * fade;
  }
  float warm = 1.0 - smoothstep(0.05, 0.35, sunDir.y);                          // low sun: warmer light
  vec3 sunCol = mix(vec3(1.0, 0.96, 0.88), vec3(1.0, 0.62, 0.38), warm) * smoothstep(0.5, 1.0, day);
  float diff = max(dot(n, sunDir), 0.0);
  // the imagery already carries its own (late-morning) relief shading: lighter model lighting on top of it
  float amb = sunMode > 0.5 ? 0.55 : sat ? 0.50 + 0.12 * n.y : 0.28 + 0.18 * n.y;
  float dk = sunMode > 0.5 ? 0.55 : sat ? 0.62 : 1.1;
  vec3 col = lin(c) * (amb * mix(0.03, 1.0, day) * vec3(0.75, 0.85, 1.0) + dk * diff * sunCol);
  vec3 v = normalize(cameraPosition - vWorld);
  if (water) col += pow(max(dot(reflect(-sunDir, vec3(0, 1, 0)), v), 0.0), 300.0) * sunCol * 2.0;
  col = pow(col, vec3(1.0 / 2.2));
  float d = length(vWorld - cameraPosition);
  float f = 1.0 - exp(-d * hazeDensity * exp(-max(min(h, cameraPosition.y), 0.0) / 6000.0));
  if (viewDist > 0.0) f = max(f, smoothstep(0.55 * viewDist, viewDist, d));
  gl_FragColor = vec4(mix(col, hazeColor * mix(0.04, 1.0, day), clamp(f, 0.0, 1.0)), 1.0);
}`;

function makeGeometry(st) {     // grid of every st-th sample (the last row / column always C); skirts on the edges
  const xs = [];
  for (let i = 0; i < C; i += st) xs.push(i);
  xs.push(C);
  const n = xs.length, nG = n * n, nV = nG + 4 * n;
  const grid = new Float32Array(nV * 2), skirt = new Float32Array(nV);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const v = j * n + i; grid[2 * v] = xs[i]; grid[2 * v + 1] = xs[j]; }
  const edges = [[], [], [], []];
  for (let t = 0; t < n; t++) {
    edges[0].push(t); edges[1].push((n - 1) * n + t); edges[2].push(t * n); edges[3].push(t * n + n - 1);
  }
  const idx = [];
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const v = j * n + i; idx.push(v, v + n, v + 1, v + 1, v + n, v + n + 1);
  }
  edges.forEach((e, k) => {
    const s0 = nG + k * n;
    e.forEach((v, t) => { grid[2 * (s0 + t)] = grid[2 * v]; grid[2 * (s0 + t) + 1] = grid[2 * v + 1]; skirt[s0 + t] = 1; });
    for (let t = 0; t < n - 1; t++) idx.push(e[t], e[t + 1], s0 + t + 1, e[t], s0 + t + 1, s0 + t);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nV * 3), 3));  // unused, required by three
  g.setAttribute('grid', new THREE.BufferAttribute(grid, 2));
  g.setAttribute('skirt', new THREE.BufferAttribute(skirt, 1));
  g.setIndex(idx);
  return g;
}
const GEOMS = { 1: makeGeometry(1), 2: makeGeometry(2), 4: makeGeometry(4) };   // 51 k / 13 k / 3.3 k vertices per chunk

// ---------------------------------------------------------------- chunk cache + LOD
const chunks = new Map();
const key = (L, r, c) => `${L}/${r}/${c}`;
function getChunk(L, r, c) {
  const k = key(L, r, c);
  let ch = chunks.get(k);
  if (!ch) {
    const s = 2 ** L / PPD;
    const [zmin, zmax] = L < 0 ? FINE[`${r}_${c}`] : [meta.levels[L].zmin[r][c], meta.levels[L].zmax[r][c]];
    const ox = xOfLon(lonOf(c * C, L)), oz = zOfLat(latOf(r * C, L)), dx = s * KX, dz = s * DEG;
    const box = new THREE.Box3(new THREE.Vector3(ox, zmin - 50, oz),
                               new THREE.Vector3(ox + C * dx, zmax + 50, oz + C * dz));
    ch = { L, r, c, k, ox, oz, dx, dz, box, size: C * dz, state: 'new', last: 0 };
    chunks.set(k, ch);
  }
  return ch;
}
const exists = (L, r, c) => L < 0 ? (FINE !== null && L >= LMIN && `${r}_${c}` in FINE)
  : L < NL && r >= 0 && c >= 0 && r < meta.levels[L].rows && c < meta.levels[L].cols;

const decodeCtx = new OffscreenCanvas(NS, NS).getContext('2d', { willReadFrequently: true });
let inflight = 0;
const MAX_INFLIGHT = MOBILE ? 6 : 12;
const BUDGET_MS = MOBILE ? 3 : 8;
async function load(ch) {
  ch.state = 'loading'; inflight++;
  try {
    const res = await fetch(`tiles/${ch.k}.${TILE_EXT}`);
    if (!res.ok) throw new Error(res.status);
    const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    decodeCtx.drawImage(bmp, 0, 0); bmp.close();
    const px = decodeCtx.getImageData(0, 0, NS, NS).data;
    const hts = new Float32Array(NS * NS);
    let tex;
    if (MOBILE && meta.built) {        // phone tiles: R*256 + G - 32768 = height, B = built-up share (build_mobile.py)
      const hb = new Float32Array(2 * NS * NS);
      for (let i = 0; i < NS * NS; i++) {
        hts[i] = hb[2 * i] = px[4 * i] * 256 + px[4 * i + 1] - 32768; hb[2 * i + 1] = px[4 * i + 2] / 255;
      }
      tex = new THREE.DataTexture(hb, NS, NS, THREE.RGFormat, THREE.FloatType);
    } else {
      for (let i = 0; i < NS * NS; i++) hts[i] = px[4 * i] * 256 + px[4 * i + 1] + px[4 * i + 2] / 256 - 32768;
      tex = new THREE.DataTexture(hts, NS, NS, THREE.RedFormat, THREE.FloatType);
    }
    tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        hmap: { value: tex }, origin: { value: new THREE.Vector2(ch.ox, ch.oz) },
        stepv: { value: new THREE.Vector2(ch.dx, ch.dz) }, skirtDepth: { value: 40 + 8 * ch.dz },
        sunDir: shared.sunDir, hazeColor: shared.hazeColor, hazeDensity: shared.hazeDensity, viewDist: shared.viewDist, day: shared.day,
        sunMode: shared.sunMode, sunTex: { value: null }, sunXf: { value: new THREE.Vector4() },
        satGlobal: shared.satGlobal, satTex: { value: null }, satOn: { value: 0 }, satXf: { value: new THREE.Vector4() },
        style: shared.style, doy: shared.doy, featTex: { value: null }, featOn: { value: 0 }, on3d: shared.on3d,
        trailTex: { value: null }, trailOn: { value: 0 }, trails: shared.trails, waterMask: shared.waterMask, featValid: { value: 0 },
      },
      vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide, defines: MOBILE ? { LITE: '' } : {},
    });
    const mesh = new THREE.Mesh(GEOMS[QS.mesh] || GEOMS[1], mat);
    mesh.frustumCulled = false; mesh.matrixAutoUpdate = false; mesh.visible = false;
    scene.add(mesh);
    Object.assign(ch, { mesh, tex, mat, heights: hts, state: 'ready' });
  } catch (e) {
    ch.state = 'missing';
  } finally { inflight--; }
}
function evict(frame) {
  if (chunks.size < (MOBILE ? 420 : 900)) return;
  const old = [...chunks.values()].filter(ch => ch.state === 'ready' && ch.L < NL - 1 && ch.last < frame - 60)
    .sort((a, b) => a.last - b.last);
  for (const ch of old.slice(0, chunks.size - (MOBILE ? 300 : 700))) {
    scene.remove(ch.mesh); ch.tex.dispose(); ch.mat.dispose(); chunks.delete(ch.k);
    if (ch.sat) ch.sat.dispose();
    if (ch.feat) ch.feat.dispose();
    if (ch.trail) ch.trail.dispose();
  }
}

const SPLIT = [1.4, 2.2, 3.5];
let splitIdx = Math.max(0, Math.min(2, QS.detail));
const frustum = new THREE.Frustum(), pv = new THREE.Matrix4();
let drawn = [];
function updateLOD(frame) {
  pv.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  frustum.setFromProjectionMatrix(pv);
  const draw = [], want = [], K = SPLIT[splitIdx], P = camera.position;
  const need = ch => { if (ch.state === 'new') want.push([ch.box.distanceToPoint(P) / ch.size, ch]); };
  const VD = QS.dist * 1000;
  function visit(ch) {
    ch.last = frame;
    if (!frustum.intersectsBox(ch.box)) return;
    if (VD && ch.box.distanceToPoint(P) > VD) return;                // beyond the view distance: not drawn, not loaded
    if (ch.state !== 'ready') { need(ch); return; }
    if (ch.L > LMIN && ch.box.distanceToPoint(P) < K * ch.size) {
      const kids = [];
      for (let dr = 0; dr < 2; dr++) for (let dc = 0; dc < 2; dc++)
        if (exists(ch.L - 1, 2 * ch.r + dr, 2 * ch.c + dc)) kids.push(getChunk(ch.L - 1, 2 * ch.r + dr, 2 * ch.c + dc));
      let ready = true;
      for (const k of kids) {
        k.last = frame;
        if (!frustum.intersectsBox(k.box) || (VD && k.box.distanceToPoint(P) > VD)) continue;
        if (k.state === 'new' || k.state === 'loading') { ready = false; need(k); }
      }
      if (ready && kids.length) { kids.forEach(visit); return; }   // no kids (outside the fine area): draw this one
    }
    draw.push(ch);
  }
  const top = meta.levels[NL - 1];
  for (let r = 0; r < top.rows; r++) for (let c = 0; c < top.cols; c++) visit(getChunk(NL - 1, r, c));
  for (const ch of drawn) ch.mesh.visible = false;
  for (const ch of draw) ch.mesh.visible = true;
  drawn = draw;
  want.sort((a, b) => a[0] - b[0]);
  for (const [, ch] of want) { if (inflight >= MAX_INFLIGHT) break; load(ch); }
  evict(frame);
}

function groundAt(x, z) {   // bilinear height from the finest loaded level, null if nothing loaded
  const lon = lonOfX(x), lat = latOfZ(z);
  for (let L = LMIN; L < NL; L++) {
    const s = 2 ** L;
    const cf = ((lon - meta.west) * PPD + 0.5) / s - 0.5, rf = ((meta.north - lat) * PPD + 0.5) / s - 0.5;
    if (cf < 0 || rf < 0) return 0;
    const col = Math.floor(cf / C), row = Math.floor(rf / C);
    const ch = chunks.get(key(L, row, col));
    if (!ch || ch.state !== 'ready') continue;
    const i = cf - col * C, j = rf - row * C;
    const i0 = Math.min(Math.floor(i), C - 1), j0 = Math.min(Math.floor(j), C - 1), fx = i - i0, fy = j - j0;
    const H = ch.heights, v = j0 * NS + i0;
    return (H[v] * (1 - fx) + H[v + 1] * fx) * (1 - fy) + (H[v + NS] * (1 - fx) + H[v + NS + 1] * fx) * fy;
  }
  return null;
}

// ---------------------------------------------------------------- sun overlay (data/sun/<run>/viewer, see serve.py)
let sunMeta = null, sunGlobal = null;
const sunTiles = new Map();
let sunInflight = 0;
async function loadRGBA(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status);
  const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const cv = new OffscreenCanvas(bmp.width, bmp.height), cx = cv.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0);
  const data = cx.getImageData(0, 0, bmp.width, bmp.height).data;
  const tex = new THREE.DataTexture(new Uint8Array(data.buffer), bmp.width, bmp.height, THREE.RGBAFormat);
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
  const out = { tex, data, w: bmp.width, h: bmp.height };
  bmp.close();
  return out;
}
(async () => {
  try {
    if (MOBILE) throw new Error('no sun overlay in the phone build');
    sunMeta = await (await fetch('sun/meta.json')).json();
    sunGlobal = await loadRGBA('sun/overview.png');
  } catch (e) { sunMeta = null; }
})();
function sunTile(r, c, frame) {
  const k = `${r}_${c}`;
  let t = sunTiles.get(k);
  if (!t) {
    t = { state: 'new', last: frame }; sunTiles.set(k, t);
  }
  t.last = frame;
  if (t.state === 'new' && sunInflight < 2) {
    t.state = 'loading'; sunInflight++;
    loadRGBA(`sun/${k}.png`).then(d => Object.assign(t, d, { state: 'ready' }))
      .catch(() => { t.state = 'missing'; }).finally(() => { sunInflight--; });
  }
  if (sunTiles.size > 30) {   // LRU
    const old = [...sunTiles.entries()].filter(([, v]) => v.state === 'ready' && v.last < frame - 30)
      .sort((a, b) => a[1].last - b[1].last);
    for (const [kk, v] of old.slice(0, sunTiles.size - 24)) { v.tex.dispose(); sunTiles.delete(kk); }
  }
  return t.state === 'ready' ? t : null;
}
// texture transform for a terrain chunk: uv = xf.xy + grid * xf.zw. A raster texel k (grouped by `group`
// cells for the overview) sits at L0 sample off + stride * (k * group + (group - 1) / 2).
function sunXform(ch, x0, y0, group, W, H, v4) {
  const f = 2 ** ch.L, st = sunMeta.stride, off = sunMeta.offset;
  const ax = ch.c * C * f + (f - 1) / 2 - x0, ay = ch.r * C * f + (f - 1) / 2 - y0;
  v4.set((((ax - off) / st - (group - 1) / 2) / group + 0.5) / W,
         (((ay - off) / st - (group - 1) / 2) / group + 0.5) / H,
         f / st / group / W, f / st / group / H);
}
function updateSun(frame) {
  if (!sunMeta || !sunGlobal || shared.sunMode.value === 0) return;
  for (const ch of drawn) {
    const u = ch.mat.uniforms;
    const t = ch.L <= 4 ? sunTile(ch.r >> (4 - ch.L), ch.c >> (4 - ch.L), frame) : null;
    if (t) {
      u.sunTex.value = t.tex;
      sunXform(ch, (ch.c >> (4 - ch.L)) * PPD, (ch.r >> (4 - ch.L)) * PPD, 1, t.w, t.h, u.sunXf.value);
    } else {
      u.sunTex.value = sunGlobal.tex;
      sunXform(ch, 0, 0, sunMeta.overview_factor, sunGlobal.w, sunGlobal.h, u.sunXf.value);
    }
  }
}
// ---------------------------------------------------------------- satellite imagery (serve.py /sat/L/r/c.jpg)
// Each chunk gets a T x T texture whose texel centres sit on its samples 0..C; until it arrives the chunk
// borrows the nearest loaded ancestor's texture (no flashing when the LOD refines).
let satInflight = 0, satFailed = 0;
const maxAniso = renderer.capabilities.getMaxAnisotropy();
async function loadSat(ch) {
  ch.satState = 'loading'; satInflight++;
  try {
    const res = await fetch(`sat/${ch.k}.jpg`);
    if (!res.ok) throw new Error(res.status);
    const bmp = await createImageBitmap(await res.blob());
    if (!chunks.has(ch.k)) { bmp.close(); return; }          // evicted meanwhile
    const t = new THREE.Texture(bmp);
    t.flipY = false; t.anisotropy = maxAniso;
    t.minFilter = THREE.LinearMipmapLinearFilter; t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping; t.needsUpdate = true;
    ch.sat = t; ch.satT = bmp.width; ch.satState = 'ready';
  } catch (e) { ch.satState = 'missing'; satFailed++; }
  finally { satInflight--; }
}
function satXform(ch, a, v4) {   // texture of ancestor (or self) a for chunk ch
  const f = 2 ** ch.L, fa = 2 ** a.L, T = a.satT;
  const ax = (ch.c * C + 0.5) * f / fa - 0.5 - a.c * C, ay = (ch.r * C + 0.5) * f / fa - 0.5 - a.r * C;
  const s = (T - 1) / (C * T), g = f / fa;
  v4.set(0.5 / T + ax * s, 0.5 / T + ay * s, g * s, g * s);
}
// ---------------------------------------------------------------- painted map: OSM features rasterised per chunk
// features.json (build_features.py) -> per chunk an RGBA mask: r river water coverage, g town (60 gravel bank, 140 inside,
// 255 wall), b cultivation (additive nearness to settlements). Drawn on the CPU with a 2-D canvas, nearest chunks first.
let FEAT = null;
const FB = 10, RB = 4;                       // settlement buckets per degree, river-piece buckets per degree
const getJson = u => fetch(u).then(r => r.ok ? r.json() : null).catch(() => null);
Promise.all([getJson('features.json'), getJson('water.json')]).then(([js, wj]) => {
  if (!js) return;
  const pts = new Map(), riv = new Map();
  const put = (m, k, v) => { let a = m.get(k); if (!a) m.set(k, a = []); a.push(v); };
  const bk = (la, lo, n) => Math.floor(la * n) * 100000 + Math.floor(lo * n);
  js.settlements.forEach(([la, lo, k, r, rc, ks]) => put(pts, bk(la, lo, FB), { la, lo, k, r, rc, ks }));
  // castles: mapped outline (flat lat/lon list) or a point -> small square keep
  js.castles.forEach(([la, lo, ring]) => {
    let r = 15;
    if (ring.length >= 6) for (let j = 0; j < ring.length; j += 2)
      r = Math.max(r, Math.hypot((ring[j] - la) * DEG, (ring[j + 1] - lo) * KX));
    put(pts, bk(la, lo, FB), { la, lo, k: 9, r, ring: ring.length >= 6 ? ring : null });
  });
  for (const [w, a] of js.rivers) {          // split into pieces of <= 40 points so each has a small bbox
    for (let i = 0; i < a.length / 2 - 1; i += 39) {
      const seg = a.slice(2 * i, 2 * Math.min(i + 40, a.length / 2));
      let la0 = 90, la1 = -90, lo0 = 180, lo1 = -180;
      for (let j = 0; j < seg.length; j += 2) {
        la0 = Math.min(la0, seg[j]); la1 = Math.max(la1, seg[j]); lo0 = Math.min(lo0, seg[j + 1]); lo1 = Math.max(lo1, seg[j + 1]);
      }
      const piece = { w, seg };
      for (let r = Math.floor(la0 * RB); r <= Math.floor(la1 * RB); r++)
        for (let c = Math.floor(lo0 * RB); c <= Math.floor(lo1 * RB); c++) put(riv, r * 100000 + c, piece);
    }
  }
  const wat = new Map();                     // water areas (build_features.py water): rings + bbox, per bucket
  if (wj) for (const rings of wj.polys) {
    let la0 = 90, la1 = -90, lo0 = 180, lo1 = -180;
    const o = rings[0];
    for (let j = 0; j < o.length; j += 2) {
      la0 = Math.min(la0, o[j]); la1 = Math.max(la1, o[j]); lo0 = Math.min(lo0, o[j + 1]); lo1 = Math.max(lo1, o[j + 1]);
    }
    const a = { rings, la0, la1, lo0, lo1 };
    for (let r = Math.floor(la0 * RB); r <= Math.floor(la1 * RB); r++)
      for (let c = Math.floor(lo0 * RB); c <= Math.floor(lo1 * RB); c++) put(wat, r * 100000 + c, a);
  }
  shared.waterMask.value = wj ? 1 : 0;
  FEAT = { pts, riv, wat, extent: (wj && wj.extent) || null };
});
function inRing(r, la, lo) {          // ray casting on a flat lat/lon ring
  let inside = false;
  for (let i = 0, j = r.length - 2; i < r.length; j = i, i += 2)
    if ((r[i] > la) !== (r[j] > la) && lo < (r[j + 1] - r[i + 1]) * (la - r[i]) / (r[j] - r[i]) + r[i + 1]) inside = !inside;
  return inside;
}
function inExtent(ch) {               // all four corners of the chunk inside the OSM extract (water data exists there)
  if (!FEAT.extent) return true;
  const W = C * ch.dx, Hm = C * ch.dz;
  for (const [x, z] of [[0, 0], [W, 0], [0, Hm], [W, Hm]]) {
    const la = latOfZ(ch.oz + z), lo = lonOfX(ch.ox + x);
    if (!FEAT.extent.some(r => inRing(r, la, lo))) return false;
  }
  return true;
}
const fhash = (a, b) => { const x = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453; return x - Math.floor(x); };
let featCanvas = null;
const REACH_M = 4000;            // margin = largest reach (cities), so fields from places outside the chunk arrive
function cultField(ch, near) {
  const W = C * ch.dx, Hm = C * ch.dz, gs = Math.max(50, Hm / 96);
  const nx = Math.ceil((W + 2 * REACH_M) / gs), nz = Math.ceil((Hm + 2 * REACH_M) / gs), n = nx * nz;
  const x0 = ch.ox - REACH_M, z0 = ch.oz - REACH_M;
  ch.cg = gs;
  const h = new Float32Array(n), cost = new Float32Array(n).fill(1e9);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const v = groundAt(x0 + (i + 0.5) * gs, z0 + (j + 0.5) * gs); h[j * nx + i] = v === null ? 0 : v;
  }
  for (const s of near) {                                   // seeds: start with the place's reach as a credit
    if (!s.rc) continue;
    const i = Math.floor((xOfLon(s.lo) - x0) / gs), j = Math.floor((zOfLat(s.la) - z0) / gs);
    if (i < 0 || j < 0 || i >= nx || j >= nz) continue;
    cost[j * nx + i] = Math.min(cost[j * nx + i], -s.rc);
  }
  // step cost: distance x (1 + 30 slope^2): 10 % slope costs 1.3x, 30 % 3.7x, 60 % 12x; lakes and sea block
  const D2 = gs * Math.SQRT2;
  const step = (a, b, d) => {
    if (h[b] <= 0.5) return 1e9;
    const sl = (h[b] - h[a]) / d; return d * (1 + 30 * sl * sl);
  };
  const relax = (k, kk, d) => { const c = cost[kk] + step(kk, k, d); if (c < cost[k]) cost[k] = c; };
  for (let it = 0; it < 3; it++) {                          // chamfer sweeps: forward then backward, 8 neighbours
    for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (i > 0) relax(k, k - 1, gs);
      if (j > 0) { relax(k, k - nx, gs); if (i > 0) relax(k, k - nx - 1, D2); if (i < nx - 1) relax(k, k - nx + 1, D2); }
    }
    for (let j = nz - 1; j >= 0; j--) for (let i = nx - 1; i >= 0; i--) {
      const k = j * nx + i;
      if (i < nx - 1) relax(k, k + 1, gs);
      if (j < nz - 1) { relax(k, k + nx, gs); if (i < nx - 1) relax(k, k + nx + 1, D2); if (i > 0) relax(k, k + nx - 1, D2); }
    }
  }
  const cv = new OffscreenCanvas(nx, nz), cx = cv.getContext('2d'), img = cx.createImageData(nx, nz);
  for (let k = 0; k < n; k++) {
    const v = Math.max(0, Math.min(1, -cost[k] / 700));     // full fields up to 700 m before the reach runs out
    img.data[4 * k + 2] = Math.round(255 * v); img.data[4 * k + 3] = 255;
  }
  cx.putImageData(img, 0, 0);
  return cv;
}
function buildFeat(ch) {
  const FT = ch.L < 0 ? 512 : ch.L === 0 ? 256 : 128;     // 6.6 / 26 / 53+ m per pixel
  if (!featCanvas || featCanvas.width !== FT) featCanvas = new OffscreenCanvas(FT, FT);
  const g = featCanvas.getContext('2d', { willReadFrequently: true });
  const W = C * ch.dx, Hm = C * ch.dz, mpx = Hm / FT;
  const X = lon => (xOfLon(lon) - ch.ox) / W * FT, Y = lat => (zOfLat(lat) - ch.oz) / Hm * FT;
  g.globalCompositeOperation = 'source-over'; g.fillStyle = '#000'; g.fillRect(0, 0, FT, FT);
  const lat1 = latOfZ(ch.oz), lat0 = latOfZ(ch.oz + Hm), lon0 = lonOfX(ch.ox), lon1 = lonOfX(ch.ox + W);
  const mla = 0.05, mlo = 0.07;              // settlements up to ~5 km outside the chunk (field reach)
  const near = [];
  for (let r = Math.floor((lat0 - mla) * FB); r <= Math.floor((lat1 + mla) * FB); r++)
    for (let c = Math.floor((lon0 - mlo) * FB); c <= Math.floor((lon1 + mlo) * FB); c++) {
      const a = FEAT.pts.get(r * 100000 + c); if (a) near.push(...a);
    }
  // farmland reach: walking cost from each settlement over the terrain (steep ground is expensive), so fields
  // spread along valley floors and gentle benches and stop at slopes; bigger places reach further (s.rc metres)
  g.globalCompositeOperation = 'lighten';
  if (!MOBILE) g.drawImage(cultField(ch, near), REACH_M / ch.cg, REACH_M / ch.cg, W / ch.cg, Hm / ch.cg, 0, 0, FT, FT);
  // rivers: gravel band, then water (max-composited so they don't add up)
  g.globalCompositeOperation = 'lighten'; g.lineCap = g.lineJoin = 'round';
  const seen = new Set();
  for (let r = Math.floor(lat0 * RB); r <= Math.floor(lat1 * RB); r++)
    for (let c = Math.floor(lon0 * RB); c <= Math.floor(lon1 * RB); c++)
      for (const pc of FEAT.riv.get(r * 100000 + c) || []) {
        if (seen.has(pc)) continue; seen.add(pc);
        // true width: no minimum, so on coarse chunks a 10 m river is a faint sub-pixel line (antialiasing =
        // coverage) instead of a band hundreds of metres wide; gravel bank in g (60, below the town threshold)
        const lw = pc.w / mpx;
        if (lw < 0.12) continue;
        g.beginPath();
        for (let j = 0; j < pc.seg.length; j += 2) g.lineTo(X(pc.seg[j + 1]), Y(pc.seg[j]));
        if (lw >= 0.5) { g.strokeStyle = 'rgb(0,60,0)'; g.lineWidth = lw * 1.8; g.stroke(); }
        g.strokeStyle = 'rgb(255,0,0)'; g.lineWidth = lw; g.stroke();
      }
  // water areas: lakes, reservoirs, river areas, filled as water (islands left out)
  g.globalCompositeOperation = 'lighten'; g.fillStyle = 'rgb(255,0,0)';
  const seenW = new Set();
  for (let r = Math.floor(lat0 * RB); r <= Math.floor(lat1 * RB); r++)
    for (let c = Math.floor(lon0 * RB); c <= Math.floor(lon1 * RB); c++)
      for (const a of FEAT.wat.get(r * 100000 + c) || []) {
        if (seenW.has(a)) continue; seenW.add(a);
        if (a.la1 < lat0 || a.la0 > lat1 || a.lo1 < lon0 || a.lo0 > lon1) continue;
        if ((a.la1 - a.la0) * DEG < 0.5 * mpx && (a.lo1 - a.lo0) * KX < 0.5 * mpx) continue;   // under half a pixel here
        g.beginPath();
        for (const ring of a.rings) {
          g.moveTo(X(ring[1]), Y(ring[0]));
          for (let j = 2; j < ring.length; j += 2) g.lineTo(X(ring[j + 1]), Y(ring[j]));
          g.closePath();
        }
        g.fill('evenodd');
      }
  // towns, villages, castles: irregular outline; towns, cities and castles get a wall
  const shape = (s, scale) => {
    g.beginPath();
    if (s.ring) {                              // castle: its mapped outline, shrunk toward the centre for the wall
      const x = X(s.lo), y = Y(s.la);
      for (let j = 0; j < s.ring.length; j += 2) g.lineTo(x + (X(s.ring[j + 1]) - x) * scale, y + (Y(s.ring[j]) - y) * scale);
      g.closePath(); return;
    }
    const nv = s.k === 9 ? 4 : s.r < 30 ? 7 : 11, x = X(s.lo), y = Y(s.la), rx = s.r / (W / FT) * scale, ry = s.r / mpx * scale;
    for (let i = 0; i < nv; i++) {
      const a = (i + 0.3 * fhash(s.la * 977, i)) / nv * 2 * Math.PI, f = 0.78 + 0.4 * fhash(s.lo * 613 + i, s.la);
      g.lineTo(x + Math.cos(a) * rx * f, y + Math.sin(a) * ry * f);
    }
    g.closePath();
  };
  const vis = s => { const x = X(s.lo), y = Y(s.la), R = 1.3 * s.r / mpx; return x > -R && y > -R && x < FT + R && y < FT + R; };
  g.fillStyle = 'rgb(0,140,0)';
  const walled = [];
  for (const s of near) {
    const rp = s.r / mpx;
    if (rp < 0.35 && s.k >= 2 && s.k !== 9) continue;                 // villages / hamlets too small here
    if (!vis(s)) continue;
    shape(s, Math.max(1, 0.7 / rp)); g.fill();                        // at least ~1.4 px across, so towns stay visible
    if ((s.k <= 1 && s.r >= 40) || s.k === 9) walled.push(s);
  }
  g.globalCompositeOperation = 'lighter'; g.strokeStyle = 'rgb(0,115,0)';
  for (const s of walled) {
    const ww = Math.max(s.k === 9 ? 4 : 6, 1.5 * mpx);
    if (s.r < 1.5 * ww) continue;
    g.lineWidth = ww / mpx; shape(s, (s.r - ww * 0.6) / s.r); g.stroke();
  }
  const img = g.getImageData(0, 0, FT, FT);
  const tex = new THREE.DataTexture(new Uint8Array(img.data.buffer), FT, FT, THREE.RGBAFormat);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
  return tex;
}
function updateFeat() {
  if (shared.style.value === 1 || !FEAT) return;                 // plain uses the rivers, painted everything
  const P = camera.position, todo = [];
  for (const ch of drawn) {
    if (ch.feat || ch.L > (MOBILE ? NL - 1 : 3)) continue;        // phone: every level (its only water source)
    todo.push([ch.box.distanceToPoint(P) / ch.size, ch]);
  }
  todo.sort((a, b) => a[0] - b[0]);
  const t0 = performance.now();
  for (const [, ch] of todo) {
    if (performance.now() - t0 > BUDGET_MS) break;               // drawing per frame (8 ms PC, 3 ms phone)
    ch.feat = buildFeat(ch);
    ch.mat.uniforms.featTex.value = ch.feat; ch.mat.uniforms.featOn.value = 1;
    ch.mat.uniforms.featValid.value = inExtent(ch) ? 1 : 0;
  }
}
// ---------------------------------------------------------------- paths & roads overlay (all map styles)
// data/trails/<layer> (build_trails.py): 0.1-degree tiles of int32 pieces (code | flags << 8 | npts << 16, then
// lat/lon in microdegrees). Layers: osm (OSM paths, code = difficulty), roads (OSM, code = road kind), ch (swisstopo
// official hiking network), euregio (digiway official trails + closures). Drawn per chunk (L <= 1) in colour into one
// premultiplied RGBA texture; what is drawn and how comes from PS (the paths menu, P) and pathStyle() below, which
// also draws the legend swatches, so map and legend cannot disagree.
const PLAYERS = ['roads', 'osm', 'ch', 'euregio'];
const pathsBox = () => document.getElementById('paths');   // ($ is defined further down)
const PL = {};
const trailTiles = new Map();
const T = (long, short) => MOBILE ? short : long;     // menu text: phones get the short version
const DIFF = [   // difficulty classes: colour, OSM dash, menu text
  ['#5a3a1e', [2, 2], 'not graded', T('OSM: no sac_scale tag · official: no grade in the data (Tyrol, South Tyrol). Difficulty unknown, check locally.', 'Difficulty unknown.')],
  ['#fbcc1f', [], 'T1 hiking', T('Yellow trail (CAI T). Well-made path, gentle, little risk of falling; trainers are fine.', 'Easy path, little fall risk.')],
  ['#e01f1a', [6, 2.5], 'T2–T3 mountain hiking', T('Red-white (CAI E/EE). Narrow, steep, exposed in places, sometimes chains; surefooted, no vertigo, boots.', 'Steep, exposed in places.')],
  ['#2966fa', [3, 1.5, 1, 1.5], 'T4–T6 alpine · via ferrata', T('Blue-white (CAI EEA). Partly pathless, scrambling with hands, snow or glacier possible; alpine experience, maybe equipment.', 'Scrambling, pathless; alpine experience.')],
];
const OFF_NOGRADE = '#ff8a1a';
const CLOSURE = '#ff1fd2';
const KINDS = ['path', 'footway', 'track', 'steps', 'bridleway', 'via ferrata'];
const ROADS = [  // name, colour, drawn width [m], min width [px]
  ['motorway', '#e5606e', 20, 2.4], ['trunk / expressway', '#f09060', 16, 2.2], ['primary', '#f8b84e', 13, 2.0],
  ['secondary', '#f0e070', 11, 1.8], ['tertiary', '#ffffff', 9, 1.6], ['minor (residential, unclassified)', '#eeeeee', 6, 1.2],
  ['service', '#d0d0d0', 4, 0.9], ['track (farm, forest)', '#a0703c', 3, 1.0],
];
const SOURCES = [
  ['osm', 'OpenStreetMap paths', T('Whole Alps, community-mapped; grade from sac_scale where tagged. Thin, dash pattern by grade; wider = regional or national route.', 'Whole Alps. Dashed by grade.')],
  ['ch', 'Official: Switzerland', T('swisstopo swissTLM3D hiking network (signposted), graded per segment. Solid line with a white edge.', 'Signposted network. White edge.')],
  ['euregio', 'Official: Tyrol · South Tyrol · Trentino', T('Euregio digiway. Graded only in Trentino, and per whole route (its hardest part). Solid line with a white edge.', 'Graded in Trentino only. White edge.')],
  ['closures', 'Trail closures', T('digiway (mainly Tyrol), notices in force when the data were built; includes cycling-only closures.', 'Tyrol; at build date.')],
  ['roads', 'Roads', T('OpenStreetMap, by kind; dark edge. Tunnels: 3-D, see below.', 'By kind.')],
];
const PS_DEF = { on: 1, src: { osm: 1, ch: 1, euregio: 1, closures: 1, roads: 1 }, cls: [1, 1, 1, 1],
                 kind: [1, 1, 1, 1, 1, 1], routes: 0, road: [1, 1, 1, 1, 1, 1, 1, 1], tun3d: 1, tunk: [1, 1, 1] };
let PS = JSON.parse(JSON.stringify(PS_DEF));
try {
  const s = JSON.parse(localStorage.getItem('flightPaths') || 'null');
  if (s) for (const k in PS) if (k in s) PS[k] = typeof PS[k] === 'object' && !Array.isArray(PS[k]) ? { ...PS[k], ...s[k] } : s[k];
} catch (e) { /* storage blocked: defaults */ }
shared.trails.value = PS.on;
let pathGen = 1;
function pathsChanged() {
  pathGen++;
  try { localStorage.setItem('flightPaths', JSON.stringify(PS)); } catch (e) { /* not remembered */ }
}
for (const n of PLAYERS) fetch(`trails/${n}/index.json`).then(r => r.ok ? r.json() : null).then(js => {
  if (!js) return;
  PL[n] = { S: js.south, W: js.west, T: js.tile, have: new Set(js.tiles), meta: js };
  pathGen++; if (!pathsBox().classList.contains('hidden')) renderPathsMenu();
}).catch(() => {});
function trailTile(k, frame) {
  let t = trailTiles.get(k);
  if (!t) {
    t = { data: null, last: frame }; trailTiles.set(k, t);
    const gz = (PL[k.split('/')[0]] || { meta: {} }).meta.gz;      // web build: gzip'd tiles (static hosts send .bin raw)
    fetch(`trails/${k}.bin${gz ? '.gz' : ''}`).then(r => r.ok ? r.arrayBuffer() : null).then(gunzip)
      .then(b => { t.data = b ? undelta(k, new Int32Array(b)) : new Int32Array(1); }).catch(() => { t.data = new Int32Array(1); });
    if (trailTiles.size > 1000) {                                  // drop the least recently used tiles
      const old = [...trailTiles.entries()].filter(([, v]) => v.data).sort((a, b) => a[1].last - b[1].last);
      for (const [kk] of old.slice(0, 250)) trailTiles.delete(kk);
    }
  }
  t.last = frame;
  return t.data;
}
async function gunzip(b) {              // gzip magic 1f 8b -> inflate (DecompressionStream: Safari 16.4+, Chrome 80+)
  if (!b || b.byteLength < 2) return b;
  const u = new Uint8Array(b, 0, 2);
  if (u[0] !== 0x1f || u[1] !== 0x8b) return b;                // already inflated by the host
  return new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
function undelta(k, d) {              // phone build: coordinates after the first of a piece are stored as differences
  const L = PL[k.split('/')[0]];
  if (!L || !L.meta.delta) return d;
  for (let p = 1, n = d[0]; n > 0; n--) {
    const np = d[p] >>> 16;
    for (let q = 1; q < np; q++) { d[p + 1 + 2 * q] += d[p - 1 + 2 * q]; d[p + 2 + 2 * q] += d[p + 2 * q]; }
    p += 1 + 2 * np;
  }
  return d;
}
function pathVisible(layer, h) {
  const c = h & 255, f = (h >> 8) & 255;
  if (layer === 'roads') return PS.road[c] && !(f & 1);        // road tunnels: 3-D layer below
  if (layer === 'osm') return PS.cls[c] && PS.kind[(f >> 2) & 7] && (!PS.routes || (f & 1));
  if (f & 128) return PS.src.closures;
  return PS.src[layer] && PS.cls[c];
}
// strokes [{z, col, w px, dash, a}] for one piece header; mpp = metres per texture pixel, lw = base path width [px]
function pathStyle(layer, h, mpp, lw) {
  const c = h & 255, f = (h >> 8) & 255;
  if (layer === 'tun') return [{ z: 0, col: c < 8 ? ROADS[c][1] : TUN_KINDS[c - 8][1], w: 2.4, dash: [4, 2.5], a: 1 }];
  if (layer === 'roads') {
    const [, col, m, mn] = ROADS[c], w = Math.max(mn, m / mpp), tun = f & 1;
    if (tun) return [{ z: 1, col, w: w * 0.6, dash: [4, 3], a: 0.5 }];
    if (c === 7) return [{ z: 1.05, col, w, dash: [3, 2], a: 0.9 }];
    return [{ z: 0, col: '#2b2b2b', w: w + 1.4, a: 0.8 }, { z: 1.9 - c * 0.1, col, w, a: 1 }];
  }
  if (layer === 'osm') {
    const w = lw * (f & 2 ? 1.5 : 1);
    return [{ z: 10 + c * 0.1, col: DIFF[c][0], w, dash: DIFF[c][1], a: (f & 32 ? 0.45 : 1) * (c ? 1 : 0.85) }];
  }
  if (f & 128) return [{ z: 30, col: '#000', w: lw * 2.4 + 1.2, a: 0.8 }, { z: 31, col: CLOSURE, w: lw * 2.4, dash: [4, 2], a: 1 }];
  const w = lw * 1.25;                                          // white edge scales with the level (thin far away)
  return [{ z: 20, col: '#fff', w: w + lw * 0.9, a: 0.85 }, { z: 21 + c * 0.1, col: c ? DIFF[c][0] : OFF_NOGRADE, w, a: 1 }];
}
function strokeAll(g, list) {           // list of [stroke, Path2D], drawn in z order
  list.sort((a, b) => a[0].z - b[0].z);
  for (const [s, p] of list) {
    g.globalAlpha = s.a; g.strokeStyle = s.col; g.lineWidth = s.w;
    g.setLineDash(s.dash || []); g.lineCap = s.dash && s.dash.length ? 'butt' : 'round';
    g.stroke(p);
  }
  g.globalAlpha = 1;
}
let trailCv = null;
function buildTrail(ch, frame) {        // texture, false (nothing here) or null (tiles still loading)
  const W = C * ch.dx, Hm = C * ch.dz;
  const lat1 = latOfZ(ch.oz), lat0 = latOfZ(ch.oz + Hm), lon0 = lonOfX(ch.ox), lon1 = lonOfX(ch.ox + W);
  const want = { roads: PS.src.roads, osm: PS.src.osm, ch: PS.src.ch, euregio: PS.src.euregio || PS.src.closures };
  const tiles = [];
  let wait = false;
  for (const n of PLAYERS) {
    const L = PL[n]; if (!L || !want[n]) continue;
    for (let i = Math.floor((lat0 - L.S) / L.T); i <= Math.floor((lat1 - L.S) / L.T); i++)
      for (let j = Math.floor((lon0 - L.W) / L.T); j <= Math.floor((lon1 - L.W) / L.T); j++) {
        const k = `${i}_${j}`; if (!L.have.has(k)) continue;
        const d = trailTile(`${n}/tiles/${k}`, frame); if (d) tiles.push([n, d]); else wait = true;
      }
  }
  if (wait) return null;
  if (!tiles.length) return false;
  const FT = ch.L < 0 ? 512 : 256;                               // 6.6 / 26 / 53 m per pixel at L -1 / 0 / 1
  const mpp = W / FT, lw = ch.L < 0 ? 1.6 : ch.L === 0 ? 1.3 : 1.0;
  if (!trailCv || trailCv.canvas.width !== FT)
    trailCv = new OffscreenCanvas(FT, FT).getContext('2d', { willReadFrequently: true });
  const g = trailCv;
  g.clearRect(0, 0, FT, FT); g.lineJoin = 'round';
  const sx = KX / W * FT * 1e-6, sy = -DEG / Hm * FT * 1e-6;
  const x0 = (LON0 * KX + ch.ox) / W * FT, y0 = (ch.oz - LAT0 * DEG) / Hm * FT;   // X = lon_u * sx - x0 ...
  const paths = new Map();                                       // layer:header -> Path2D (null = hidden)
  for (const [n, d] of tiles) {
    for (let p = 1, cnt = d[0]; cnt > 0; cnt--) {
      const h = d[p] & 0xFFFF, np = d[p] >>> 16, key = n + h;
      let P = paths.get(key);
      if (P === undefined) { P = pathVisible(n, h) ? new Path2D() : null; paths.set(key, P); }
      if (P) {
        P.moveTo(d[p + 2] * sx - x0, d[p + 1] * sy - y0);
        for (let q = 1; q < np; q++) P.lineTo(d[p + 2 + 2 * q] * sx - x0, d[p + 1 + 2 * q] * sy - y0);
      }
      p += 1 + 2 * np;
    }
  }
  const list = [];
  for (const [key, P] of paths) {
    if (!P) continue;
    const n = key.replace(/\d+$/, ''), h = +key.slice(n.length);
    for (const s of pathStyle(n, h, mpp, lw)) list.push([s, P]);
  }
  if (!list.length) return false;
  strokeAll(g, list);
  const D = g.getImageData(0, 0, FT, FT).data;
  for (let k = 0; k < D.length; k += 4) {                         // premultiply: correct mipmaps and blending
    const a = D[k + 3] / 255; D[k] *= a; D[k + 1] *= a; D[k + 2] *= a;
  }
  const tex = new THREE.DataTexture(new Uint8Array(D.buffer), FT, FT, THREE.RGBAFormat);
  tex.generateMipmaps = true; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = maxAniso; tex.needsUpdate = true;
  return tex;
}
function updateTrails(frame) {
  if (!shared.trails.value) return;
  const P = camera.position, todo = [];
  for (const ch of drawn) {
    if (ch.trailGen === pathGen || ch.L > 1) continue;
    todo.push([ch.box.distanceToPoint(P) / ch.size, ch]);
  }
  todo.sort((a, b) => a[0] - b[0]);
  const t0 = performance.now();
  for (const [, ch] of todo) {
    if (performance.now() - t0 > BUDGET_MS) break;
    const t = buildTrail(ch, frame);
    if (t === null) continue;
    if (ch.trail) ch.trail.dispose();                            // the old texture stays until its replacement is ready
    ch.trail = t; ch.trailGen = pathGen;
    ch.mat.uniforms.trailTex.value = t || null; ch.mat.uniforms.trailOn.value = t ? 1 : 0;
  }
}
// ---- tunnels in 3-D (build_trails.py tunnels): real OSM alignment, heights interpolated between the portals.
// Two passes over the same lines: dashed and see-through where the terrain is in front (inside the mountain), solid
// where they are in view. Rebuilt for tunnels within TUN_R of the camera when it moves or the menu changes.
const TUN_KINDS = [['railway', '#c86bff'], ['metro / light rail / tram', '#8fa0ff'], ['funicular / rack railway', '#ff9df0']];
const TUN_R = 45000;
const tunMats = [
  new LineMaterial({ linewidth: 2.5, vertexColors: true, dashed: true, dashSize: 50, gapSize: 35, transparent: true,
                     opacity: 0.8, depthWrite: false, depthFunc: THREE.GreaterDepth }),
  new LineMaterial({ linewidth: 3.5, vertexColors: true }),
];
tunMats.forEach(m => m.resolution.set(innerWidth, innerHeight));
let TUN = null, tunLoading = false, tunObj = null, tunAt = null, tunGen = -1;
function updateTunnels() {
  const want = shared.trails.value > 0 && PS.tun3d + PS.tunk.reduce((a, b) => a + b, 0) > 0;
  if (tunObj) tunObj.forEach(o => { o.visible = want; });
  if (!want) return;
  if (!TUN) {
    if (!tunLoading) {
      tunLoading = true;
      fetch('trails/tunnels/tunnels.json').then(r => r.ok ? r.json() : null).then(js => {
        TUN = { built: js && js.built, list: [] };
        if (js) for (const [k, name, f] of js.tunnels) {
          const n = f.length / 3, xyz = new Float32Array(3 * n);
          let x0 = 1e9, x1 = -1e9, z0 = 1e9, z1 = -1e9;
          for (let i = 0; i < n; i++) {
            const x = xOfLon(f[3 * i + 1]), z = zOfLat(f[3 * i]);
            xyz[3 * i] = x; xyz[3 * i + 1] = f[3 * i + 2]; xyz[3 * i + 2] = z;
            x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
          }
          TUN.list.push({ k, name, xyz, cx: (x0 + x1) / 2, cz: (z0 + z1) / 2, r: Math.hypot(x1 - x0, z1 - z0) / 2 });
        }
      }).catch(() => { TUN = { list: [] }; });
    }
    return;
  }
  const P = camera.position;
  if (tunObj && tunGen === pathGen && Math.hypot(P.x - tunAt.x, P.z - tunAt.z) < TUN_R / 5) return tunSight();
  tunGen = pathGen; tunAt = { x: P.x, z: P.z };
  if (tunObj) { scene.remove(...tunObj); tunObj[0].geometry.dispose(); tunObj = null; }
  const pos = [], col = [], c = new THREE.Color();
  for (const t of TUN.list) {
    if (Math.hypot(t.cx - P.x, t.cz - P.z) - t.r > TUN_R) continue;
    if (t.k < 8 ? !(PS.tun3d && PS.road[t.k]) : !PS.tunk[t.k - 8]) continue;
    c.set(t.k < 8 ? ROADS[t.k][1] : TUN_KINDS[t.k - 8][1]);
    const a = t.xyz;
    for (let i = 0; i + 5 < a.length; i += 3) {
      // pieces of <= 400 m, each shown only while the ground above it is in sight (tunSight)
      const n = Math.max(1, Math.ceil(Math.hypot(a[i + 3] - a[i], a[i + 5] - a[i + 2]) / 400));
      for (let q = 0; q < n; q++) {
        const f0 = q / n, f1 = (q + 1) / n;
        for (const f of [f0, f1]) pos.push(a[i] + (a[i + 3] - a[i]) * f, a[i + 1] + (a[i + 4] - a[i + 1]) * f, a[i + 2] + (a[i + 5] - a[i + 2]) * f);
        col.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
    }
  }
  if (!pos.length) return;
  const geo = new LineSegmentsGeometry();
  geo.setPositions(new Float32Array(pos)); geo.setColors(new Float32Array(col));
  tunObj = tunMats.map((m, i) => {
    const o = new LineSegments2(geo, m);
    o.computeLineDistances(); o.frustumCulled = false; o.renderOrder = 10 + i; o.visible = want;
    return o;
  });
  scene.add(...tunObj);
  // every piece starts hidden (collapsed far below the terrain) and appears once its check says it is in sight
  const buf = geo.attributes.instanceStart.data;
  tunSeg = { full: Float32Array.from(buf.array), vis: new Uint8Array(pos.length / 6), buf, next: 0 };
  for (let k = 0; k < tunSeg.vis.length; k++) hideSeg(k);
  buf.needsUpdate = true;
  tunSight(Math.min(tunSeg.vis.length, 4000));                   // most of them right away
}
let tunSeg = null;
const tunPt = { x: 0, y: 0, z: 0 };
function hideSeg(k) { const a = tunSeg.buf.array; for (let j = 0; j < 6; j++) a[6 * k + j] = j % 3 === 1 ? -1e5 : tunSeg.full[6 * k + (j % 3)]; }
function tunSight(n = MOBILE ? 80 : 300) {        // line of sight from the camera to the ground above each piece
  if (!tunSeg) return;
  const S = tunSeg, F = S.full, a = S.buf.array, P = camera.position, VD = QS.dist * 1000 || TUN_R;
  let changed = false;
  for (let c = 0; c < Math.min(n, S.vis.length); c++) {
    const k = S.next; S.next = (S.next + 1) % S.vis.length;
    const x = (F[6 * k] + F[6 * k + 3]) / 2, z = (F[6 * k + 2] + F[6 * k + 5]) / 2, g = groundAt(x, z);
    let v = 0;
    if (g !== null && Math.hypot(x - P.x, z - P.z) < Math.min(VD, TUN_R)) {
      tunPt.x = x; tunPt.y = g + 20; tunPt.z = z;
      v = occluded(tunPt) ? 0 : 1;
    }
    if (v === S.vis[k]) continue;
    S.vis[k] = v; changed = true;
    if (v) for (let j = 0; j < 6; j++) a[6 * k + j] = F[6 * k + j]; else hideSeg(k);
  }
  if (changed) S.buf.needsUpdate = true;
}
// ---- the paths & roads menu (P): sources, difficulty, kinds, roads, each with its legend swatch
function swatch(cv, layer, h) {
  const g = cv.getContext('2d'), w = cv.width, y = cv.height / 2;
  g.fillStyle = '#7f8c76'; g.fillRect(0, 0, w, cv.height);       // a neutral terrain colour behind the line
  const p = new Path2D(); p.moveTo(3, y); p.lineTo(w - 3, y);
  strokeAll(g, pathStyle(layer, h, 2.5, 4.5).map(s => [{ ...s, w: Math.max(s.w, s.z ? 4 : 5.4),   // 2x canvas
    dash: s.dash && s.dash.map(v => v * 3) }, p]));
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
function renderPathsMenu() {
  const box = pathsBox(), cb = (g, i, v, dis) => `<input type="checkbox" data-g="${g}" data-i="${i}"${v ? ' checked' : ''}${dis ? ' disabled' : ''}>`;
  const sw = (layer, h) => `<canvas class="sw" width="88" height="28" data-sw="${layer}:${h}"></canvas>`;
  const have = n => n === 'closures' ? PL.euregio : PL[n];
  let s = `<div class="ph"><b>Paths &amp; roads</b><span class="x" data-close="1" title="close (P)">×</span></div>
    <label>${cb('on', 0, PS.on)} show on the map <span class="k">(Shift+P)</span></label>
    <h4>Sources</h4>`;
  for (const [n, name, d] of SOURCES) {
    const L = have(n), when = L ? (n === 'closures' ? L.meta.closures_date : L.meta.built) : null;
    s += `<label class="${L ? '' : 'off'}">${cb('src', n, PS.src[n], !L)} ${name}</label>
      <div class="d">${esc(d)}${L ? (MOBILE ? '' : ` <span class="k">(data ${esc(when || '?')})</span>`) : ' <b>not built</b> (build_trails.py)'}</div>`;
  }
  s += `<h4>Difficulty <span class="k">OSM · official</span></h4>`;
  DIFF.forEach(([, , name, d], c) => {
    s += `<label>${cb('cls', c, PS.cls[c])} ${sw('osm', c)} ${sw('ch', c)} ${name}</label><div class="d">${esc(d)}</div>`;
  });
  s += `<label>${sw('euregio', 128 << 8)} closed (official closure notice)</label>
    <h4>Kinds of OSM path</h4><div class="grid">`;
  KINDS.forEach((k, i) => { s += `<label>${cb('kind', i, PS.kind[i])} ${k}</label>`; });
  s += `</div><label>${cb('routes', 0, PS.routes)} only paths on signed hiking routes</label>
    ${MOBILE ? '' : '<div class="d">footway, track and steps are drawn only when graded or on a hiking route; tunnels faint.</div>'}
    <h4>Roads</h4>`;
  ROADS.forEach(([name], i) => { s += `<label>${cb('road', i, PS.road[i])} ${sw('roads', i)} ${name}</label>`; });
  s += `<h4>Tunnels <span class="k">3-D, dashed inside the mountain</span></h4>
    <label>${cb('tun3d', 0, PS.tun3d)} ${sw('tun', 0)} road tunnels${MOBILE ? '' : ' <span class="k">(road colours, follow the road kinds)</span>'}</label>`;
  TUN_KINDS.forEach(([name], i) => { s += `<label>${cb('tunk', i, PS.tunk[i])} ${sw('tun', 8 + i)} ${name}</label>`; });
  s += MOBILE ? `<div class="d">Within ${TUN_R / 1000} km. Heights estimated between the portals.</div>
    <div class="note">OSM grades are often missing or too low. Where lines disagree, assume the harder grade.
    No snow or current conditions.</div>` : `<div class="d">Within ${TUN_R / 1000} km of you. Real alignment from OSM; heights interpolated between the portals
    (GLO-30), so the summit point of a long base tunnel is not reproduced (Gotthard: straight 460 → 312 m, real apex 549 m).
    ${TUN && TUN.built ? `(data ${esc(TUN.built)})` : ''}</div>
    <div class="note">Safety: OSM grades are often missing or lower than the official marking. In Switzerland
    12.6 % of graded km are shown easier, and a third of the blue-white alpine routes appear red (T2–T3); in Trentino
    official grades apply to whole routes. Where official and OSM lines disagree, assume the harder grade. None of these
    data show snow, rockfall or current conditions.</div>`;
  box.innerHTML = s;
  box.querySelectorAll('canvas[data-sw]').forEach(cv => { const [l, h] = cv.dataset.sw.split(':'); swatch(cv, l, +h); });
}
pathsBox().addEventListener('change', e => {
  const t = e.target, g = t.dataset.g, i = t.dataset.i, v = t.checked ? 1 : 0;
  if (!g) return;
  if (g === 'on') { PS.on = v; shared.trails.value = v; }
  else if (Array.isArray(PS[g])) PS[g][+i] = v;
  else if (typeof PS[g] === 'object') PS[g][i] = v;
  else PS[g] = v;
  pathsChanged(); t.blur();
});
pathsBox().addEventListener('click', e => { if (e.target.dataset.close) togglePathsMenu(); });
function togglePathsMenu() {
  const box = pathsBox(), open = box.classList.toggle('hidden') === false;
  if (open) { renderPathsMenu(); if (locked) document.exitPointerLock(); }
}
// ---------------------------------------------------------------- painted map: medieval towns and castles in 3-D
// Built in the browser from features.json for places within R3D of the camera, in 0.05-degree cells, as instanced
// meshes (houses: walls + gable roofs; churches; town walls with towers; castles along their mapped outline with
// towers and a keep). Placed on the loaded terrain (L <= 0), so a cell waits until fine heights are there.
const R3D = 6000, R3D_DROP = 9000, CELL = 0.05;
let b3dOn = !MOBILE;
const cells3d = new Map();
const b3d = new THREE.Group(); scene.add(b3d);
const sunLight = new THREE.DirectionalLight(0xffffff, 2.2), skyLight = new THREE.HemisphereLight(0xbfd4ff, 0x6b5a45, 0.9);
scene.add(sunLight, sunLight.target, skyLight);
const GEO = (() => {
  const box = new THREE.BoxGeometry(1, 1, 1); box.translate(0, 0.5, 0);
  // gable roof: footprint [-0.5, 0.5]^2, ridge along z at x = 0, height 1
  // counter-clockwise seen from outside (front faces); the first version was wound inward and got culled
  const v = [-0.5, 0, -0.5, 0, 1, 0.5, 0, 1, -0.5, -0.5, 0, -0.5, -0.5, 0, 0.5, 0, 1, 0.5,
             0.5, 0, 0.5, 0, 1, -0.5, 0, 1, 0.5, 0.5, 0, 0.5, 0.5, 0, -0.5, 0, 1, -0.5,
             -0.5, 0, 0.5, 0.5, 0, 0.5, 0, 1, 0.5, 0.5, 0, -0.5, -0.5, 0, -0.5, 0, 1, -0.5];
  const gable = new THREE.BufferGeometry(); gable.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  gable.computeVertexNormals();
  const pyr = new THREE.ConeGeometry(Math.SQRT1_2, 1, 4, 1); pyr.rotateY(Math.PI / 4); pyr.translate(0, 0.5, 0);
  const cyl = new THREE.CylinderGeometry(0.5, 0.5, 1, 12); cyl.translate(0, 0.5, 0);
  const cone = new THREE.ConeGeometry(0.62, 1, 12); cone.translate(0, 0.5, 0);
  return { box, gable, pyr, cyl, cone };
})();
const MAT3D = new THREE.MeshLambertMaterial({ color: 0xffffff });
function groundFine(x, z) {      // height from a loaded L <= 0 chunk, else null (wait for it)
  const lon = lonOfX(x), lat = latOfZ(z);
  for (let L = LMIN; L <= 0; L++) {
    const s = 2 ** L;
    const cf = ((lon - meta.west) * PPD + 0.5) / s - 0.5, rf = ((meta.north - lat) * PPD + 0.5) / s - 0.5;
    const col = Math.floor(cf / C), row = Math.floor(rf / C);
    const ch = chunks.get(key(L, row, col));
    if (!ch || ch.state !== 'ready') continue;
    const i = cf - col * C, j = rf - row * C;
    const i0 = Math.min(Math.floor(i), C - 1), j0 = Math.min(Math.floor(j), C - 1), fx = i - i0, fy = j - j0;
    const H = ch.heights, q = j0 * NS + i0;
    return (H[q] * (1 - fx) + H[q + 1] * fx) * (1 - fy) + (H[q + NS] * (1 - fx) + H[q + NS + 1] * fx) * fy;
  }
  return null;
}
function townOutline(s) {        // same irregular polygon as the 2-D mask (buildFeat), in world x/z
  const cx = xOfLon(s.lo), cz = zOfLat(s.la), nv = s.r < 30 ? 7 : 11, out = [];
  for (let i = 0; i < nv; i++) {
    const a = (i + 0.3 * fhash(s.la * 977, i)) / nv * 2 * Math.PI, f = 0.78 + 0.4 * fhash(s.lo * 613 + i, s.la);
    out.push([cx + Math.cos(a) * s.r * f, cz + Math.sin(a) * s.r * f]);
  }
  return out;
}
function inPoly(P, x, z) {
  let inside = false;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++)
    if ((P[i][1] > z) !== (P[j][1] > z) && x < (P[j][0] - P[i][0]) * (z - P[i][1]) / (P[j][1] - P[i][1]) + P[i][0]) inside = !inside;
  return inside;
}
const C3 = (r, g, b) => new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
const WALLC = [C3(0.86, 0.83, 0.75), C3(0.80, 0.74, 0.62), C3(0.66, 0.62, 0.55), C3(0.90, 0.88, 0.82)];
const WOODC = [C3(0.46, 0.34, 0.23), C3(0.55, 0.42, 0.29)];
const ROOF_S = [C3(0.70, 0.36, 0.24), C3(0.62, 0.32, 0.22), C3(0.76, 0.46, 0.30)];
const ROOF_N = [C3(0.62, 0.33, 0.23), C3(0.52, 0.49, 0.46), C3(0.43, 0.34, 0.27), C3(0.37, 0.38, 0.41)];
const STONE = C3(0.68, 0.64, 0.57), SLATE = C3(0.32, 0.33, 0.36);
// one cell: collect parts as [geometry key, x, y, z, sx, sy, sz, yaw, color], then one InstancedMesh per geometry
function buildCell(ci, cj) {
  const parts = [];
  const lat0 = ci * CELL, lon0 = cj * CELL;
  const list = [];
  for (let r = Math.floor(lat0 * FB); r <= Math.floor((lat0 + CELL) * FB); r++)
    for (let c = Math.floor(lon0 * FB); c <= Math.floor((lon0 + CELL) * FB); c++)
      for (const s of FEAT.pts.get(r * 100000 + c) || [])
        if (s.la >= lat0 && s.la < lat0 + CELL && s.lo >= lon0 && s.lo < lon0 + CELL) list.push(s);
  // footprint on the ground: base at the lowest corner, walls reach down below it on slopes
  const place = (x, z, w, d, yaw) => {
    const c = Math.cos(yaw), sn = Math.sin(yaw);
    let lo = Infinity, hi = -Infinity;
    for (const [u, v] of [[0, 0], [-w / 2, -d / 2], [w / 2, -d / 2], [-w / 2, d / 2], [w / 2, d / 2]]) {
      const g = groundFine(x + u * c + v * sn, z - u * sn + v * c);
      if (g === null) return null;
      lo = Math.min(lo, g); hi = Math.max(hi, g);
    }
    return [lo - 0.5, hi - lo + 0.5];
  };
  const house = (x, z, w, d, h, yaw, wall, roof, pitch = 0.55) => {
    const p = place(x, z, w, d, yaw); if (!p) return false;
    const [y0, extra] = p;
    parts.push(['box', x, y0, z, w, h + extra, d, yaw, wall]);
    parts.push(['gable', x, y0 + h + extra, z, w * 1.08, w * pitch, d * 1.04, yaw, roof]);
    return true;
  };
  const tower = (x, z, w, h, round, wall, roof) => {
    const p = place(x, z, w, w, 0); if (!p) return false;
    const [y0, extra] = p;
    parts.push([round ? 'cyl' : 'box', x, y0, z, w, h + extra, w, 0, wall]);
    parts.push([round ? 'cone' : 'pyr', x, y0 + h + extra, z, w * 1.15, w * (round ? 1.3 : 0.9), w * 1.15, 0, roof]);
    return true;
  };
  const wallRun = (P, height, thick, towerEvery, towerW, round, roof) => {   // closed polygon of [x, z]
    for (let i = 0; i < P.length; i++) {
      const [x0, z0] = P[i], [x1, z1] = P[(i + 1) % P.length];
      const L = Math.hypot(x1 - x0, z1 - z0); if (L < 1) continue;
      const n = Math.max(1, Math.ceil(L / 25)), yaw = Math.atan2(x1 - x0, z1 - z0);
      for (let k = 0; k < n; k++) {
        const t = (k + 0.5) / n, x = x0 + (x1 - x0) * t, z = z0 + (z1 - z0) * t;
        const p = place(x, z, thick, L / n, yaw); if (!p) return false;
        parts.push(['box', x, p[0], z, thick, height + p[1], L / n + 0.3, yaw, STONE]);
      }
      const nt = Math.max(1, Math.round(L / towerEvery));
      for (let k = 0; k < nt; k++) {
        const t = k / nt;
        if (!tower(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, towerW, height + 5, round, STONE, roof)) return false;
      }
    }
    return true;
  };
  for (const s of list) {
    const cx = xOfLon(s.lo), cz = zOfLat(s.la), seed = s.la * 7919 + s.lo * 104729;
    const rnd = (() => { let k = 0; return () => fhash(seed, ++k); })();
    const hgt = groundFine(cx, cz); if (hgt === null) return null;              // terrain not fine enough yet
    const south = s.la < 46.3 + 0.15 * (rnd() - 0.5);
    const alpine = hgt > 1000 && !south;
    const roofC = () => alpine && rnd() < 0.6 ? WOODC[1] : south ? ROOF_S[Math.floor(rnd() * 3)] : ROOF_N[Math.floor(rnd() * 4)];
    const wallC = () => alpine && rnd() < 0.6 ? WOODC[Math.floor(rnd() * 2)] : WALLC[Math.floor(rnd() * 4)];
    if (s.k === 9) {                                      // castle: walls along the outline, towers, keep, hall
      let P;
      if (s.ring) {
        P = []; for (let j = 0; j < s.ring.length; j += 2) P.push([xOfLon(s.ring[j + 1]), zOfLat(s.ring[j])]);
        if (P.length > 2 && Math.hypot(P[0][0] - P.at(-1)[0], P[0][1] - P.at(-1)[1]) < 1) P.pop();
      } else {
        const a = rnd() * Math.PI, w = 14;
        P = [[-w, -w], [w, -w], [w, w], [-w, w]].map(([u, v]) => [cx + u * Math.cos(a) - v * Math.sin(a), cz + u * Math.sin(a) + v * Math.cos(a)]);
      }
      const big = s.r > 60;
      if (!wallRun(P, big ? 10 : 8, 2.6, big ? 45 : 30, big ? 7 : 5.5, true, rnd() < 0.5 ? SLATE : ROOF_N[0])) return null;
      let kx = 0, kz = 0; P.forEach(([x, z]) => { kx += x; kz += z; }); kx /= P.length; kz /= P.length;
      if (!inPoly(P, kx, kz)) [kx, kz] = [cx, cz];
      const kw = big ? 13 : 9;
      if (!tower(kx, kz, kw, big ? 28 : 20, false, STONE, SLATE)) return null;
      if (s.r > 25) house(kx + kw * 1.4, kz, 9, 18, 9, rnd() * Math.PI, STONE, ROOF_N[Math.floor(rnd() * 4)]);
      continue;
    }
    const P = townOutline(s);
    const walled = s.k <= 1 && s.r >= 40;
    if (s.r < 32) {                                       // hamlet / small village: a few scattered houses
      const n = Math.max(1, Math.round(Math.PI * s.r * s.r / 160));
      for (let i = 0; i < n; i++) {
        const a = rnd() * 2 * Math.PI, d = Math.sqrt(rnd()) * s.r * 0.9;
        house(cx + Math.cos(a) * d, cz + Math.sin(a) * d, 7 + 2 * rnd(), 9 + 4 * rnd(), 3.5 + 2.5 * rnd(), rnd() * Math.PI, wallC(), roofC());
      }
      if (s.k === 2 && rnd() < 0.7) {                     // village church
        const a = rnd() * Math.PI, ox = cx + 14 * Math.cos(a), oz = cz + 14 * Math.sin(a);
        house(ox, oz, 8, 16, 7, a, WALLC[3], south ? ROOF_S[0] : SLATE, 0.7);
        tower(ox + 7 * Math.sin(a), oz + 7 * Math.cos(a), 4.5, 18, false, WALLC[3], south ? ROOF_S[0] : SLATE);
      }
      continue;
    }
    // town: organic plan. Winding main streets run from the market square to the edge, broken ring lanes cross
    // them, short alleys branch off; houses line both sides facing the street, the rest fills with backyard
    // buildings. Overlaps are avoided with a coarse occupancy grid of circles.
    const th = fhash(s.la * 31, s.lo * 17) * Math.PI, ct = Math.cos(th), st = Math.sin(th);
    const sq = Math.max(16, 0.15 * s.r);
    const inner = walled ? P.map(([px, pz]) => [cx + (px - cx) * 0.92, cz + (pz - cz) * 0.92]) : P;
    const occ = new Map(), OC = 8;
    const okAt = (x, z, rad) => {
      const gi = Math.floor(x / OC), gj = Math.floor(z / OC), m = Math.ceil(rad / OC) + 1;
      for (let a = gi - m; a <= gi + m; a++) for (let b = gj - m; b <= gj + m; b++)
        for (const [ox, oz, or] of occ.get(a * 1e6 + b) || []) if (Math.hypot(ox - x, oz - z) < or + rad) return false;
      return true;
    };
    const take = (x, z, rad) => { const k = Math.floor(x / OC) * 1e6 + Math.floor(z / OC); (occ.get(k) || occ.set(k, []).get(k)).push([x, z, rad]); };
    const streets = [];                                    // [points [[x, z]], half width]
    const walk = (x, z, a, len, turn, hw) => {
      const pts = [[x, z]];
      for (let d = 0; d < len; d += 10) {
        a += (rnd() - 0.5) * turn; x += 10 * Math.sin(a); z += 10 * Math.cos(a);
        if (!inPoly(P, x, z)) break;
        pts.push([x, z]);
      }
      if (pts.length > 1) streets.push([pts, hw]);
      return pts;
    };
    const nMain = s.k === 0 ? 6 + Math.floor(rnd() * 3) : s.k === 1 ? 4 + Math.floor(rnd() * 2) : 3;
    const mains = [];
    for (let i = 0; i < nMain; i++) {
      const a = th + (i + 0.35 * (rnd() - 0.5)) * 2 * Math.PI / nMain;
      mains.push(walk(cx + sq * Math.sin(a), cz + sq * Math.cos(a), a, 1.3 * s.r, 0.45, 3.5));
    }
    for (const fr of s.r > 70 ? [0.42, 0.72] : s.r > 45 ? [0.6] : []) {   // ring lanes, with gaps
      const ph = rnd() * 6.28, rr = fr * s.r, pts = [];
      for (let a = 0; a < 2 * Math.PI; a += 10 / rr) {
        const w = rr * (1 + 0.12 * Math.sin(3 * a + ph) + 0.05 * (rnd() - 0.5));
        const x = cx + w * Math.sin(a), z = cz + w * Math.cos(a);
        if (rnd() < 0.04 && pts.length > 2) { streets.push([pts.splice(0), 2.5]); continue; }
        pts.push([x, z]);
      }
      if (pts.length > 1) streets.push([pts, 2.5]);
    }
    for (const m of mains) for (let i = 2; i < m.length - 1; i += 3 + Math.floor(rnd() * 3)) {   // alleys
      const [x0, z0] = m[i], [x1, z1] = m[i + 1], a = Math.atan2(x1 - x0, z1 - z0) + (rnd() < 0.5 ? 1 : -1) * (1.2 + 0.7 * rnd());
      walk(x0, z0, a, 25 + 45 * rnd(), 0.6, 1.8);
    }
    take(cx, cz, sq);                                       // market square stays open
    const cs = s.k === 0 ? 1.8 : s.k === 1 ? 1.3 : 1.0;     // church (cathedral in cities) on the square
    const ox = cx + (sq + 6 * cs) * st, oz = cz + (sq + 6 * cs) * ct;
    take(ox, oz, 14 * cs); take(ox + 15 * cs * st, oz + 15 * cs * ct, 5 * cs);
    for (const [pts, hw] of streets) for (const [x, z] of pts) take(x, z, hw);
    const storeysOf = () => s.k === 0 ? 2 + Math.floor(rnd() * 3) : s.k === 1 ? 2 + Math.floor(rnd() * 2) : 1 + Math.floor(rnd() * 2);
    const tryHouse = (x, z, w, d, yaw) => {
      const rad = 0.42 * Math.hypot(w, d);
      if (!inPoly(inner, x, z) || !okAt(x, z, rad)) return true;
      take(x, z, rad);
      return house(x, z, w, d, 1.2 + storeysOf() * 2.9, yaw, wallC(), roofC());
    };
    // street frontage: gable ends to the street, set back by the half width
    for (const [pts, hw] of streets) for (let i = 0; i < pts.length - 1; i++) {
      const [x0, z0] = pts[i], [x1, z1] = pts[i + 1], L = Math.hypot(x1 - x0, z1 - z0);
      if (L < 1) continue;
      const ux = (x1 - x0) / L, uz = (z1 - z0) / L, yaw = Math.atan2(ux, uz);
      for (let t = 0; t < L; t += 8 + 3 * rnd()) for (const side of [-1, 1]) {
        if (rnd() < 0.12) continue;
        const w = 6.5 + 2.5 * rnd(), d = 9 + 4 * rnd(), off = hw + d / 2 + 0.6 + rnd();
        const x = x0 + ux * t + side * uz * off, z = z0 + uz * t - side * ux * off;
        if (!tryHouse(x, z, w, d, yaw + Math.PI / 2 + (rnd() - 0.5) * 0.25)) return null;
      }
    }
    // backyards: smaller buildings in what is left, any orientation
    const nFill = Math.round(Math.PI * s.r * s.r / 260);
    for (let i = 0; i < nFill; i++) {
      const a = rnd() * 2 * Math.PI, d = Math.sqrt(rnd()) * s.r;
      if (!tryHouse(cx + d * Math.sin(a), cz + d * Math.cos(a), 5 + 3 * rnd(), 7 + 4 * rnd(), rnd() * Math.PI)) return null;
    }
    house(ox, oz, 11 * cs, 26 * cs, 11 * cs, th, WALLC[3], south ? ROOF_S[0] : SLATE, 0.75);
    tower(ox + 15 * cs * st, oz + 15 * cs * ct, 6 * cs, 28 * cs, false, WALLC[3], south ? ROOF_S[0] : SLATE);
    if (walled && !wallRun(P, s.k === 0 ? 10 : 8, 2.2, 55, 6, s.k === 0, south ? ROOF_S[0] : SLATE)) return null;
  }
  // instanced meshes, one per geometry
  const grp = new THREE.Group(), m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(),
        pos = new THREE.Vector3(), sc = new THREE.Vector3();
  const byGeo = {};
  for (const p of parts) (byGeo[p[0]] = byGeo[p[0]] || []).push(p);
  for (const [g, ps] of Object.entries(byGeo)) {
    const mesh = new THREE.InstancedMesh(GEO[g], MAT3D, ps.length);
    ps.forEach((p, i) => {
      pos.set(p[1], p[2], p[3]); sc.set(p[4], p[5], p[6]); q.setFromEuler(e.set(0, p[7], 0));
      mesh.setMatrixAt(i, m4.compose(pos, q, sc)); mesh.setColorAt(i, p[8]);
    });
    mesh.computeBoundingSphere();
    grp.add(mesh);
  }
  grp.userData.n = parts.length;
  return grp;
}
function update3d() {
  const on = b3dOn && shared.style.value === 2 && FEAT;
  b3d.visible = !!on; shared.on3d.value = on ? 1 : 0;
  if (!on) return;
  const P = camera.position, lat = latOfZ(P.z), lon = lonOfX(P.x);
  const dla = R3D / DEG, dlo = R3D / KX;
  for (const [k, c] of cells3d) {                         // drop far cells
    if (Math.hypot((c.lat - lat) * DEG, (c.lon - lon) * KX) > R3D_DROP) {
      if (c.grp) { b3d.remove(c.grp); c.grp.children.forEach(m => m.dispose()); }
      cells3d.delete(k);
    }
  }
  const want = [];
  for (let i = Math.floor((lat - dla) / CELL); i <= Math.floor((lat + dla) / CELL); i++)
    for (let j = Math.floor((lon - dlo) / CELL); j <= Math.floor((lon + dlo) / CELL); j++) {
      const k = i * 100000 + j;
      if (cells3d.has(k)) continue;
      const clat = (i + 0.5) * CELL, clon = (j + 0.5) * CELL, d = Math.hypot((clat - lat) * DEG, (clon - lon) * KX);
      if (d < R3D + 3000) want.push([d, i, j, k, clat, clon]);
    }
  want.sort((a, b) => a[0] - b[0]);
  const t0 = performance.now();
  for (const [, i, j, k, clat, clon] of want) {
    if (performance.now() - t0 > 12) break;              // ~12 ms of building per frame
    const grp = buildCell(i, j);
    if (!grp) continue;                                   // terrain not loaded finely enough yet: retry later
    b3d.add(grp); cells3d.set(k, { grp, lat: clat, lon: clon });
  }
}
function light3d() {             // lights follow the simulated sun; buildings sit in the same twilight as the terrain
  const d = shared.day.value, s = shared.sunDir.value;
  sunLight.position.copy(camera.position).addScaledVector(s, 1000); sunLight.target.position.copy(camera.position);
  const warm = 1 - Math.min(1, Math.max(0, (s.y - 0.05) / 0.3));
  sunLight.color.setRGB(1, 0.96 - 0.34 * warm, 0.88 - 0.5 * warm);
  sunLight.intensity = 2.4 * Math.min(1, Math.max(0, (d - 0.5) * 2));
  skyLight.intensity = 0.9 * (0.03 + 0.97 * d);
}
function updateSat() {
  if (!shared.satGlobal.value) return;
  const P = camera.position, want = [];
  for (const ch of drawn) {
    const u = ch.mat.uniforms;
    if (!ch.satState) want.push([ch.box.distanceToPoint(P) / ch.size, ch]);
    let a = ch;
    while (a && a.satState !== 'ready') a = a.L + 1 < NL ? chunks.get(key(a.L + 1, a.r >> 1, a.c >> 1)) : null;
    if (a) { u.satTex.value = a.sat; u.satOn.value = 1; satXform(ch, a, u.satXf.value); }
    else u.satOn.value = 0;
  }
  want.sort((x, y) => x[0] - y[0]);
  for (const [, ch] of want) { if (satInflight >= 6) break; loadSat(ch); }
}

function sunAt(x, z) {   // [hours, sunless days] at a world position, or null
  if (!sunMeta) return null;
  const X = (lonOfX(x) - meta.west) * PPD, Y = (meta.north - latOfZ(z)) * PPD;
  const r = Math.floor(Y / PPD), c = Math.floor(X / PPD);
  const t = sunTiles.get(`${r}_${c}`);
  if (!t || t.state !== 'ready') return null;
  const i = Math.round((X - c * PPD - sunMeta.offset) / sunMeta.stride), j = Math.round((Y - r * PPD - sunMeta.offset) / sunMeta.stride);
  if (i < 0 || j < 0 || i >= t.w || j >= t.h) return null;
  const p = 4 * (j * t.w + i);
  if (t.data[p + 2] < 128) return 'sea';
  return [t.data[p] * sunMeta.hours_scale, t.data[p + 1]];
}

// ---------------------------------------------------------------- place names (build_labels.py, OpenStreetMap)
// Shown within a distance that grows with importance, hidden behind terrain (line of sight sampled on the
// loaded heights), decluttered on screen by priority (importance / distance).
let labels = [], labelsOn = true, labelsPlaces = false;   // N: all names -> places only -> off
const labelLayer = document.getElementById('labels'), labelPool = [];
function maxDistKm(k, r) {
  switch (k) {
    case 'city': return r >= 500000 ? 300 : r >= 100000 ? 150 : r >= 30000 ? 60 : r >= 10000 ? 30 : 14;
    case 'peak': return r >= 4000 ? 120 : r >= 3500 ? 60 : r >= 3000 ? 35 : r >= 2500 ? 18 : r >= 2000 ? 10 : 5;
    case 'lake': return r >= 50 ? 200 : r >= 10 ? 90 : r >= 2 ? 40 : 15;
    case 'river': return r >= 10 ? 70 : r >= 4 ? 35 : 15;
    default: return r >= 20 ? 150 : r >= 5 ? 45 : 12;         // valley: major / normal / tiny
  }
}
const valleyClass = r => (r >= 20 ? 'major' : r >= 5 ? 'normal' : 'tiny');   // length along the valley, km
const KIND_W = { city: 3, peak: 2.2, lake: 2, river: 1.2, valley: 1 };
const VALLEY = { major: { w: 3, lift: 250, px: 15 }, normal: { w: 1.4, lift: 80, px: 8.8 }, tiny: { w: 0.8, lift: 15, px: 6.2 } };
fetch('labels.json').then(r => r.json()).then(js => {
  // peaks carry a 7th field, isolation = km to the nearest higher named peak: below 3 km they are "minor",
  // drawn small and only within 1.2 + 1.2 * isolation km (~1.5-4.8 km), so they don't clutter the view
  labels = js.labels.map(([kind, name, lat, lon, z, rank, iso]) => {
    const vc = kind === 'valley' ? VALLEY[valleyClass(rank)] : null;
    const minor = kind === 'peak' && iso !== undefined && iso < 3;
    return {
      kind, name, rank, x: xOfLon(lon), z: zOfLat(lat),
      y: z + (vc ? vc.lift : kind === 'peak' ? 15 : kind === 'city' ? 30 : 10),
      dmax: (minor ? Math.min(maxDistKm(kind, rank), 1.2 + 1.2 * iso) : maxDistKm(kind, rank)) * 1000, vis: true, chk: -1e9,
      w: vc ? vc.w : minor ? 0.9 : KIND_W[kind], px: vc ? vc.px : minor ? 6.2 : 7.5,
      cls: 'lbl ' + kind + (vc ? ' v-' + valleyClass(rank) : '') + (minor ? ' minor' : ''),
      html: kind === 'peak' ? `▲ ${esc(name)} <small>${Math.round(rank)}</small>` : kind === 'city' ? `● ${esc(name)}` : esc(name),   // OSM text: never as HTML
    };
  });
}).catch(() => {});
function occluded(L) {
  const P = camera.position, n = 24;
  for (let i = 1; i < n; i++) {
    const t = i / n, x = P.x + (L.x - P.x) * t, z = P.z + (L.z - P.z) * t, y = P.y + (L.y - P.y) * t;
    const g = groundAt(x, z);
    if (g !== null && g > y + 3) return true;
  }
  return false;
}
const lv = new THREE.Vector3();
let labelNear = [], labelNearAt = null, labelNearT = 0;
const NAME_MODES = ['all names', 'places only', 'no names'];
function namesMode() { return !labelsOn ? 2 : labelsPlaces ? 1 : 0; }
function cycleNames() {
  const m = (namesMode() + 1) % 3;
  labelsOn = m < 2; labelsPlaces = m === 1;
  if (!labelsOn) labelPool.forEach(d => { d.style.display = 'none'; });
  toast(NAME_MODES[m]);
  const b = document.querySelector('[data-set="names"]'); if (b) b.textContent = `Names: ${NAME_MODES[m]}`;
}
function updateLabels(frame) {
  if (!labelsOn || !labels.length) return;
  const P = camera.position, W = innerWidth, H = innerHeight, cand = [];
  const now = performance.now(), VD = QS.dist * 1000 || 1e12;
  // labels within reach (+ 3 km margin): rescanned only every second or after 2 km of movement (100 k labels)
  if (!labelNearAt || now - labelNearT > 1000 || Math.hypot(P.x - labelNearAt.x, P.z - labelNearAt.z) > 2000) {
    labelNear = labels.filter(L => Math.hypot(L.x - P.x, L.z - P.z) < Math.min(L.dmax, VD) + 3000);
    labelNearAt = { x: P.x, z: P.z }; labelNearT = now;
  }
  let checks = 0;
  for (const L of labelNear) {
    const d = Math.hypot(L.x - P.x, L.y - P.y, L.z - P.z);
    if (d > L.dmax || d > VD || (labelsPlaces && L.kind !== 'city')) continue;
    lv.set(L.x, L.y, L.z).project(camera);
    if (lv.z > 1 || lv.x < -1.1 || lv.x > 1.1 || lv.y < -1.1 || lv.y > 1.1) continue;
    if (now - L.chk > 400 && checks < (MOBILE ? 60 : 250)) { L.vis = !occluded(L); L.chk = now + Math.random() * 200; checks++; }
    if (!L.vis) continue;
    cand.push([L.w * L.dmax / Math.max(d, 300), L, (lv.x + 1) / 2 * W, (1 - lv.y) / 2 * H]);
  }
  cand.sort((a, b) => b[0] - a[0]);
  const placed = [];
  let n = 0;
  for (const [, L, sx, sy] of cand) {
    if (n >= QS.labelsMax) break;
    const w = L.px * (L.name.length + (L.kind === 'peak' ? 7 : 2)), h = L.px * 2.1;
    const box = [sx - w / 2, sy - h, sx + w / 2, sy + 2];
    if (placed.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
    placed.push(box);
    let el = labelPool[n];
    if (!el) { el = document.createElement('div'); labelLayer.appendChild(el); labelPool.push(el); }
    if (el._L !== L) { el._L = L; el.className = L.cls; el.innerHTML = L.html; }
    el.style.display = 'block';
    el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) translate(-50%, -100%)`;
    n++;
  }
  for (let i = n; i < labelPool.length; i++) if (labelPool[i].style.display !== 'none') labelPool[i].style.display = 'none';
}

// ---------------------------------------------------------------- controls
let yaw = 0, pitch = 0, flySpeed = 2500, walkSpeed = 6, mode = 'fly', vy = 0, locked = false;
let level = true;   // fly: true = W/S keep the altitude (Space up, Ctrl/C down); false = fly where you look (V)
const keys = new Set();
const $ = id => document.getElementById(id);

function teleport(lat, lon, elev, bearing = 180) {
  const tx = xOfLon(lon), tz = zOfLat(lat), b = bearing * Math.PI / 180, D = 6000;
  const cx = tx + D * Math.sin(b), cz = tz - D * Math.cos(b), cy = elev + (elev > 2500 ? 700 : 1500);
  camera.position.set(cx, cy, cz);
  const dx = tx - cx, dz = tz - cz;
  yaw = Math.atan2(-dx, -dz); pitch = Math.atan2(elev - cy, Math.hypot(dx, dz));
  if (mode === 'walk') setMode('fly');
}
function setMode(m) { mode = m; vy = 0; $('h-mode').textContent = m === 'fly' ? (level ? 'fly · level' : 'fly · free') : m; }
function setStyle(s) {          // map style; imagery is fetched only while it is shown
  shared.style.value = s; shared.satGlobal.value = s === 1 ? 1 : 0;
  $('h-style').textContent = STYLES[s];
  try { localStorage.setItem('flightStyle', STYLES[s]); } catch (e) { /* storage blocked: not remembered */ }
}
{
  let s = new URLSearchParams(location.search).get('style');
  if (!s) try { s = localStorage.getItem('flightStyle'); } catch (e) { s = null; }
  setStyle(MOBILE ? 0 : Math.max(0, STYLES.indexOf(s || 'satellite')));
}
function setSunMode(m) {
  if (m > 0 && !sunMeta) { $('h-sun').textContent = 'no sun data (run scripts/sun_alps.py)'; m = 0; }
  shared.sunMode.value = m;
  $('legend').classList.toggle('hidden', m === 0);
  if (m === 1) {
    $('lg-title').textContent = `direct sun per year ${sunMeta.year} (${sunMeta.res_m} m)`;
    $('lg-bar').style.background = 'linear-gradient(90deg,#08051a,#521270,#b83578,#fa785c,#fcf2b8)';
    $('lg-lo').textContent = '0 h'; $('lg-hi').textContent = '4500 h';
  } else if (m === 2) {
    $('lg-title').textContent = `days without direct sun ${sunMeta.year}`;
    $('lg-bar').style.background = 'linear-gradient(90deg,#d9d9d9,#9ec7eb,#082666)';
    $('lg-lo').textContent = '0'; $('lg-hi').textContent = '150+ days';
  }
}

const sel = $('places');
PLACES.forEach((p, i) => { const o = document.createElement('option'); o.value = i; o.textContent = p[0]; sel.appendChild(o); });
sel.addEventListener('change', () => {
  if (sel.value !== '') { const p = PLACES[+sel.value]; teleport(p[1], p[2], p[3], p[4]); }
  sel.value = ''; sel.blur();
});

canvas.addEventListener('click', () => { if (!MOBILE) canvas.requestPointerLock(); });
document.addEventListener('pointerlockchange', () => {
  locked = document.pointerLockElement === canvas;
  $('start').classList.toggle('hidden', locked);
  $('cross').style.display = locked ? 'block' : 'none';
  if (!locked) keys.clear();
});
document.addEventListener('mousemove', e => {
  if (!locked) return;
  yaw -= e.movementX * 0.0022;
  pitch = Math.max(-1.55, Math.min(1.55, pitch - e.movementY * 0.0022));
});
// speed steps (- / =), each with a comparison; Shift multiplies (fly x20, walk x5)
const FLY_STEPS = [[5, 'cyclist'], [15, 'car in town'], [30, 'car on the motorway'], [60, 'helicopter'],
  [120, 'light aircraft'], [250, 'airliner'], [500, 'fighter jet'], [1000, 'rifle bullet'], [2500, 'default'],
  [5000, 'fast overview'], [10000, 'crossing the Alps in ~1 min'], [30000, 'crossing the Alps in ~20 s'], [100000, 'max']];
const WALK_STEPS = [[0.8, 'stroll'], [1.3, 'hiking'], [2, 'brisk walk'], [3, 'jog'], [4.5, 'run'], [6, 'default'],
  [10, 'sprint'], [20, 'skiing'], [40, 'downhill skiing'], [100, 'max']];
const kmh = v => v * 3.6 >= 100 ? (v * 3.6).toFixed(0) : (v * 3.6).toFixed(1);
function speedStep() {
  const steps = mode === 'fly' ? FLY_STEPS : WALK_STEPS, v = mode === 'fly' ? flySpeed : walkSpeed;
  let k = 0;
  steps.forEach(([s], i) => { if (Math.abs(Math.log(s / v)) < Math.abs(Math.log(steps[k][0] / v))) k = i; });
  return [steps, k];
}
function speedText() {
  const [steps, k] = speedStep(), [v, what] = steps[k];
  return `${v} m/s = ${kmh(v)} km/h · ${what}  (${k + 1}/${steps.length})`;
}
let toastTimer = 0;
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.remove('hidden');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add('hidden'), 1800);
}
function changeSpeed(d) {
  const [steps, k] = speedStep(), v = steps[Math.max(0, Math.min(steps.length - 1, k + d))][0];
  if (mode === 'fly') flySpeed = v; else walkSpeed = v;
  toast(`${mode === 'fly' ? 'flying' : 'walking'} speed ${speedText()}   · Shift ×${mode === 'fly' ? 20 : 5}`);
}
addEventListener('keydown', e => {
  if (document.activeElement === sel) return;
  keys.add(e.code);
  if (e.code === 'Space' || e.code === 'Period' || e.code.startsWith('Arrow')) e.preventDefault();
  if (e.repeat) return;
  switch (e.code) {
    case 'Minus': case 'NumpadSubtract': changeSpeed(-1); break;           // speed steps
    case 'Equal': case 'NumpadAdd': changeSpeed(1); break;
    case 'KeyG': if (!MOBILE) setMode(mode === 'fly' ? 'walk' : 'fly'); break;   // phone: flying only
    case 'KeyV': level = !level; setMode('fly'); break;
    case 'KeyM': $('mini').classList.toggle('hidden'); break;
    case 'KeyH': ['hud', 'help', 'mini'].forEach(id => $(id).classList.toggle('hidden')); break;
    case 'KeyF': hazeIdx = (hazeIdx + 1) % HAZE.length; shared.hazeDensity.value = HAZE[hazeIdx]; break;
    case 'KeyL': setQ('detail', (splitIdx + 1) % SPLIT.length); break;
    case 'KeyX': toggleQualityMenu(); break;                        // quality menu
    case 'KeyO': setSunMode((shared.sunMode.value + 1) % 3); break;
    case 'BracketLeft': simTime -= 3600e3; break;            // clock -1 h / +1 h
    case 'BracketRight': simTime += 3600e3; break;
    case 'Digit9': simTime -= (e.shiftKey ? 30 : 1) * 86400e3; break;   // -1 day (Shift: -30 days)
    case 'Digit0': simTime += (e.shiftKey ? 30 : 1) * 86400e3; break;
    case 'KeyK': simTime = Date.now(); break;                 // back to now
    case 'KeyT': if (!MOBILE) setStyle((shared.style.value + 1) % 3); break;     // plain -> satellite -> painted
    case 'KeyB': b3dOn = !b3dOn; break;                          // 3-D towns and castles (painted map)
    case 'KeyP':                                                  // paths & roads menu; Shift+P overlay on/off
      if (e.shiftKey) {
        PS.on = shared.trails.value = 1 - shared.trails.value; pathsChanged();
        if (!pathsBox().classList.contains('hidden')) renderPathsMenu();
      } else togglePathsMenu();
      break;
    case 'KeyN': cycleNames(); break;
  }
});
addEventListener('keyup', e => keys.delete(e.code));

// ---------------------------------------------------------------- quality: view distance, depth range, resolution
let autoT = 0;
function applyQuality(dt) {
  const VD = QS.dist * 1000;
  // without the log depth buffer the near plane follows the height above ground (depth precision far away)
  const g = groundAt(camera.position.x, camera.position.z), agl = g === null ? 1000 : camera.position.y - g;
  const near = QS.logDepth ? 1 : Math.max(0.5, Math.min(200, agl * 0.04)), far = VD ? VD * 1.15 : 900000;
  if (Math.abs(camera.near - near) > 0.1 * near || camera.far !== far) {
    camera.near = near; camera.far = far; camera.updateProjectionMatrix();
  }
  shared.viewDist.value = VD;                                       // the edge fades into the haze, no hard cut
  if (QS.res === 'auto' && (autoT += dt) > 2 && fps > 0) {       // hold ~30 fps by the resolution
    autoT = 0;
    const pr = fps < 26 ? Math.max(0.5, pixelRatio - 0.125) : fps > 45 ? Math.min(PR_MAX, pixelRatio + 0.125) : pixelRatio;
    if (pr !== pixelRatio) { pixelRatio = pr; renderer.setPixelRatio(pr); }
  }
  if (autoT === 0 || frame % 30 === 0) qNow();
}
const Q_OPTS = {
  dist: [[8, '8 km'], [15, '15 km'], [25, '25 km'], [40, '40 km'], [60, '60 km'], [100, '100 km'], [0, MOBILE ? '∞' : 'unlimited']],
  mesh: MOBILE ? [[1, 'full'], [2, 'half'], [4, 'quarter']] : [[1, 'full (30 m)'], [2, 'half (60 m)'], [4, 'quarter (120 m)']],
  res: [['auto', MOBILE ? 'auto' : 'auto (~30 fps)'], [0.5, '50 %'], [0.75, '75 %'], [1, '100 %'], [1.5, MOBILE ? '150 %' : 'sharp (150 %)']],
  detail: [[0, 'low'], [1, 'medium'], [2, 'high']],
  labelsMax: [[0, 'none'], [20, '20'], [40, '40'], [80, '80'], [140, '140']],
  logDepth: [[0, MOBILE ? 'fast' : 'fast (phones)'], [1, 'precise']],
  aa: [[1, 'on'], [0, 'off']],
};
const Q_TEXT = MOBILE ? {
  dist: ['View distance', 'Biggest speed-up.'], mesh: ['Terrain mesh', 'Half looks the same, 4x faster.'],
  res: ['Resolution', 'Auto keeps ~30 fps.'], detail: ['Detail', 'When near tiles get finer.'],
  labelsMax: ['Names', 'Most names at once.'], logDepth: ['Depth', 'Precise: no far flicker, slower.'],
  aa: ['Smooth edges', 'Cheap on most phones.'],
} : {
  dist: ['View distance', 'Nothing beyond is drawn or loaded; haze fades the edge. The biggest speed-up.'],
  mesh: ['Terrain mesh', 'Triangles per tile; shading keeps the full 30 m detail. Half = 4x fewer vertices.'],
  res: ['Resolution', 'Pixels rendered (relative to the screen). Auto lowers it when the frame rate drops.'],
  detail: ['Detail switch', 'How early nearby tiles switch to finer levels (L on the keyboard).'],
  labelsMax: ['Names on screen', 'Most names shown at once (N toggles them).'],
  logDepth: ['Depth buffer', 'Precise avoids flicker of far ridges but is slow on phones; switching reloads the view.'],
  aa: ['Edge smoothing', 'Antialiasing of ridge lines; cheap on most phones. Switching reloads the view.'],
};
function renderQualityMenu() {
  let h = `<div class="ph"><b>Quality</b><span class="x" data-close="1">×</span></div>
    <div class="d" id="q-now" style="margin-left:0"></div>`;
  for (const k of Object.keys(Q_OPTS)) {
    h += `<h4>${Q_TEXT[k][0]}</h4><div class="qrow">` + Q_OPTS[k].map(([v, t]) =>
      `<button data-q="${k}" data-v="${v}" class="${String(QS[k]) === String(v) ? 'on' : ''}">${String(QS[k]) === String(v) ? '✓ ' : ''}${t}</button>`).join('') +
      `</div><div class="d" style="margin-left:0">${Q_TEXT[k][1]}</div>`;
  }
  h += `<button data-reset="1" class="qreset">defaults</button>`;
  const box = $('quality'), top = box.scrollTop;                 // redrawn on every tap: keep the scroll position
  box.innerHTML = h;
  box.scrollTop = top;
  qNow();
}
function qNow() {
  const el = $('q-now');
  if (el) el.textContent = `now ${fps.toFixed(0)} fps · ${drawn.length} tiles · resolution ${Math.round(pixelRatio * 100)} %`;
}
function setQ(k, v) {
  const was = QS[k];
  QS[k] = v; saveQS();
  if ((k === 'logDepth' || k === 'aa') && was !== v) { location.reload(); return; }   // renderer options: reload
  if (k === 'mesh') for (const ch of chunks.values()) if (ch.mesh) ch.mesh.geometry = GEOMS[v];
  if (k === 'res' && v !== 'auto') { pixelRatio = Math.min(PR_MAX, v); renderer.setPixelRatio(pixelRatio); }
  if (k === 'detail') splitIdx = v;
}
function toggleQualityMenu() {
  const box = $('quality'), open = box.classList.toggle('hidden') === false;
  if (open) { renderQualityMenu(); if (locked) document.exitPointerLock(); }
}
$('quality').addEventListener('click', e => {
  const t = e.target;
  if (t.dataset.close) return toggleQualityMenu();
  if (t.dataset.reset) { for (const k in QS_DEF) setQ(k, QS_DEF[k]); return renderQualityMenu(); }
  if (!t.dataset.q) return;
  const k = t.dataset.q, raw = t.dataset.v, v = raw === 'auto' ? raw : +raw;
  setQ(k, v); renderQualityMenu();
});

// ---------------------------------------------------------------- touch controls (phone build)
// left: move stick (proportional); drag anywhere else on the view: look; buttons send the same codes as the keys
const joy = { f: 0, r: 0, u: 0 };                               // stick forward / right, height slider
if (MOBILE) {
  document.body.classList.add('mobile');
  for (const ev of ['gesturestart', 'gesturechange', 'gestureend']) document.addEventListener(ev, e => e.preventDefault());
  $('mini').classList.add('hidden');                             // narrow screen: the map button shows it
  $('h-set').previousElementSibling.textContent = 'speed setting';     // no − = keys on a phone
  for (const id of ['h-style', 'h-lod', 'h-time', 'h-sunpos', 'h-sun', 'h-hdg', 'h-gnd'])
    if ($(id)) $(id).parentElement.style.display = 'none';
  // four round buttons on the right: height (spring slider), speed (step slider), map, settings (sub-menu)
  const ui = document.createElement('div');
  ui.innerHTML = `<div id="joy"><div id="knob"></div></div>
    <div id="rail">
      <button class="rb" data-pop="height"><b>⇕</b><small>height</small></button>
      <button class="rb" data-pop="speed"><b>»</b><small>speed</small></button>
      <button class="rb" data-pop="time"><b>◐</b><small>time</small></button>
      <button class="rb" data-act="map"><b>▦</b><small>map</small></button>
      <button class="rb" data-pop="settings"><b>⚙</b><small>settings</small></button>
    </div>
    <div id="pop-height" class="pop hidden"><div class="plabel">up</div>
      <div class="vtrack" id="htrack"><div class="vthumb" id="hthumb"></div></div><div class="plabel">down</div></div>
    <div id="pop-speed" class="pop hidden"><div class="plabel" id="spdlbl"></div>
      <div class="vtrack" id="strack"><div class="vthumb" id="sthumb"></div></div><div class="plabel">slower</div></div>
    <div id="pop-time" class="pop hidden">
      <div class="plabel" id="tlbl"></div>
      <label class="hs">time of day<input type="range" id="tday" min="0" max="1439" step="5"></label>
      <label class="hs">date<input type="range" id="tdate" min="0" max="364" step="1"></label>
      <button id="tnow">now</button>
    </div>
    <div id="pop-settings" class="pop hidden">
      <button data-set="paths">Paths &amp; roads</button>
      <button data-set="names">Names: ${NAME_MODES[namesMode()]}</button>
      <button data-set="info">Info panel</button>
      <button data-set="quality">Quality</button>
      <button data-set="level">Flight: ${level ? 'level' : 'free'}</button>
      <div class="credits">Terrain: Copernicus GLO-30 · Map data © OpenStreetMap contributors · Towns: GHSL
        © European Union, CC BY 4.0 · Trails: swisstopo, Euregio digiway</div>
    </div>`;
  document.body.appendChild(ui);
  const pops = ['height', 'speed', 'time', 'settings'];
  const openPop = name => pops.forEach(n => {
    const on = n === name && $('pop-' + n).classList.contains('hidden');
    $('pop-' + n).classList.toggle('hidden', !on);
    document.querySelector(`[data-pop="${n}"]`).classList.toggle('on', on);
    if (on && n === 'speed') showSpeed();
    if (on && n === 'time') showTime();
  });
  ui.querySelectorAll('[data-pop]').forEach(b => b.addEventListener('pointerdown', e => { e.preventDefault(); openPop(b.dataset.pop); }));
  ui.querySelector('[data-act="map"]').addEventListener('pointerdown', e => {
    e.preventDefault(); $('mini').classList.toggle('hidden');
  });
  ui.querySelectorAll('[data-set]').forEach(b => b.addEventListener('click', () => {
    const k = b.dataset.set;
    if (k === 'paths') { openPop(null); togglePathsMenu(); }
    else if (k === 'quality') { openPop(null); toggleQualityMenu(); }
    else if (k === 'names') cycleNames();
    else if (k === 'info') $('hud').classList.toggle('hidden');
    else if (k === 'level') { level = !level; setMode('fly'); b.textContent = `Flight: ${level ? 'level' : 'free'}`; }
  }));
  // time: sliders for the time of day and the date (Swiss time, like the info panel); the clock keeps running from there
  const zoneOff = t => {                                         // Europe/Zurich offset from UTC [ms] at time t
    const d = new Date(t), z = new Date(d.toLocaleString('en-US', { timeZone: 'Europe/Zurich' }));
    return z - new Date(d.toLocaleString('en-US', { timeZone: 'UTC' }));
  };
  const local = () => simTime + zoneOff(simTime);
  const showTime = () => {
    const L = local(), d = new Date(L), tod = Math.floor((L % 86400e3) / 60e3);
    $('tday').value = tod;
    $('tdate').value = Math.floor((L - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400e3);
    $('tlbl').textContent = fmtTime(simTime);
  };
  $('tday').addEventListener('input', e => {
    const L = local(), day = Math.floor(L / 86400e3) * 86400e3;
    simTime = day + e.target.value * 60e3 - zoneOff(simTime); $('tlbl').textContent = fmtTime(simTime);
  });
  $('tdate').addEventListener('input', e => {
    const L = local(), y = new Date(L).getUTCFullYear(), tod = L % 86400e3;
    simTime = Date.UTC(y, 0, 1) + e.target.value * 86400e3 + tod - zoneOff(simTime); $('tlbl').textContent = fmtTime(simTime);
  });
  $('tnow').addEventListener('click', () => { simTime = Date.now(); showTime(); });
  // a vertical slider: pointer -> value -1 (bottom) .. 1 (top)
  const slider = (track, thumb, onMove, onEnd) => {
    let id = null;
    const val = e => { const r = track.getBoundingClientRect(); return Math.max(-1, Math.min(1, 1 - 2 * (e.clientY - r.top - 26) / (r.height - 52))); };
    track.addEventListener('pointerdown', e => { id = e.pointerId; try { track.setPointerCapture(id); } catch (err) { /* */ } onMove(val(e)); });
    track.addEventListener('pointermove', e => { if (e.pointerId === id) onMove(val(e)); });
    const end = e => { if (e.pointerId !== id) return; id = null; if (onEnd) onEnd(); };
    track.addEventListener('pointerup', end); track.addEventListener('pointercancel', end);
  };
  const place = (track, thumb, v) => { thumb.style.top = `calc(26px + (100% - 52px) * ${(1 - v) / 2})`; };   // knob stays inside
  // height: climb / sink rate while held, springs back to 0
  const ht = $('htrack'), hth = $('hthumb');
  place(ht, hth, 0);
  slider(ht, hth, v => { joy.u = Math.abs(v) < 0.08 ? 0 : v; place(ht, hth, v); }, () => { joy.u = 0; place(ht, hth, 0); });
  // speed: the named steps from bottom (slow) to top (fast)
  const st = $('strack'), sth = $('sthumb');
  const showSpeed = () => {
    const [steps, k] = speedStep();
    place(st, sth, steps.length > 1 ? -1 + 2 * k / (steps.length - 1) : 0);
    $('spdlbl').innerHTML = `${steps[k][0]} m/s<br><small>${kmh(steps[k][0])} km/h · ${steps[k][1]}</small>`;
  };
  slider(st, sth, v => {
    const k = Math.round((v + 1) / 2 * (FLY_STEPS.length - 1));
    flySpeed = FLY_STEPS[k][0]; showSpeed();
  });
  const J = $('joy'), knob = $('knob');
  let jid = null;
  const jmove = e => {
    const r = J.getBoundingClientRect(), R = r.width / 2;
    let dx = (e.clientX - r.left - R) / R, dy = (e.clientY - r.top - R) / R;
    const L = Math.hypot(dx, dy); if (L > 1) { dx /= L; dy /= L; }
    joy.r = dx; joy.f = -dy;
    knob.style.transform = `translate(${dx * R * 0.7}px, ${dy * R * 0.7}px)`;
  };
  J.addEventListener('pointerdown', e => { jid = e.pointerId; try { J.setPointerCapture(jid); } catch (err) { /* */ } jmove(e); });
  J.addEventListener('pointermove', e => { if (e.pointerId === jid) jmove(e); });
  const jend = e => { if (e.pointerId !== jid) return; jid = null; joy.f = joy.r = 0; knob.style.transform = ''; };
  J.addEventListener('pointerup', jend); J.addEventListener('pointercancel', jend);
  // one finger: look; two fingers: spread = fly forward where you look, pinch = back (step ~ height above ground)
  const look = new Map();                                         // pointer id -> last position
  let pinch = null;
  const spread = () => { const [a, b] = [...look.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]); };
  canvas.addEventListener('pointerdown', e => {
    look.set(e.pointerId, [e.clientX, e.clientY]); try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* */ }
    if (look.size === 2) pinch = { d: spread() };
  });
  canvas.addEventListener('pointermove', e => {
    const p = look.get(e.pointerId); if (!p) return;
    if (pinch && look.size >= 2) {
      look.set(e.pointerId, [e.clientX, e.clientY]);
      const d = Math.max(spread(), 1), g = groundAt(camera.position.x, camera.position.z);
      const agl = g === null ? 1000 : Math.max(camera.position.y - g, 2);
      camera.getWorldDirection(mv);                                 // view direction (pitch included)
      camera.position.addScaledVector(mv, Math.log(d / pinch.d) * 2 * (agl + 30));
      if (g !== null && camera.position.y < g + 1.8) camera.position.y = g + 1.8;
      pinch.d = d;
      return;
    }
    const k = 0.005;
    yaw -= (e.clientX - p[0]) * k;
    pitch = Math.max(-1.55, Math.min(1.55, pitch - (e.clientY - p[1]) * k));
    look.set(e.pointerId, [e.clientX, e.clientY]);
  });
  const lend = e => { look.delete(e.pointerId); if (look.size < 2) pinch = null; };
  canvas.addEventListener('pointerup', lend); canvas.addEventListener('pointercancel', lend);
}
addEventListener('blur', () => keys.clear());

// ---------------------------------------------------------------- minimap
const mini = $('minimap'), mctx = mini.getContext('2d');
const MW = 320, MH = Math.round(MW * ((meta.north - meta.south) * DEG) / ((meta.east - meta.west) * KX));
mini.width = MW * devicePixelRatio; mini.height = MH * devicePixelRatio;   // CSS sets the width, height follows
const ovImg = new Image(); ovImg.src = 'tiles/overview.png';
function drawMini() {
  const s = devicePixelRatio;
  mctx.setTransform(s, 0, 0, s, 0, 0);
  if (ovImg.complete) mctx.drawImage(ovImg, 0, 0, MW, MH); else { mctx.fillStyle = '#234'; mctx.fillRect(0, 0, MW, MH); }
  const lon = lonOfX(camera.position.x), lat = latOfZ(camera.position.z);
  const px = (lon - meta.west) / (meta.east - meta.west) * MW, py = (meta.north - lat) / (meta.north - meta.south) * MH;
  mctx.save(); mctx.translate(px, py); mctx.rotate(-yaw);
  mctx.fillStyle = '#ff3b30'; mctx.strokeStyle = '#fff'; mctx.lineWidth = 1.2;
  mctx.beginPath(); mctx.moveTo(0, -9); mctx.lineTo(5, 6); mctx.lineTo(0, 3); mctx.lineTo(-5, 6); mctx.closePath();
  mctx.fill(); mctx.stroke(); mctx.restore();
}
mini.addEventListener('click', e => {
  const r = mini.getBoundingClientRect();
  const lon = meta.west + (e.clientX - r.left) / r.width * (meta.east - meta.west);
  const lat = meta.north - (e.clientY - r.top) / r.height * (meta.north - meta.south);
  camera.position.set(xOfLon(lon), Math.max(camera.position.y, 3500), zOfLat(lat));
  if (mode === 'walk') setMode('fly');
});

// ---------------------------------------------------------------- start position (URL hash or Matterhorn)
(function () {
  const h = location.hash.slice(1).split(',').map(Number);
  if (h.length === 5 && h.every(Number.isFinite)) {
    camera.position.set(xOfLon(h[1]), h[2], zOfLat(h[0]));
    yaw = -h[3] * Math.PI / 180; pitch = h[4] * Math.PI / 180;
  } else teleport(...PLACES[0].slice(1));
  setMode('fly');
})();

// ---------------------------------------------------------------- loop
const clock = new THREE.Clock();
const fwd = new THREE.Vector3(), right = new THREE.Vector3(), mv = new THREE.Vector3(), prev = new THREE.Vector3();
let frame = 0, fpsT = 0, fpsN = 0, fps = 0, hudT = 0, hashT = 0, measSpeed = 0;
function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.1);
  frame++;
  prev.copy(camera.position);
  const k = c => (keys.has(c) ? 1 : 0);
  const boost = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 20 : 1;
  // keyboard look (works when the mouse is blocked, e.g. touchpad disabled while typing)
  yaw += (k('ArrowLeft') - k('ArrowRight')) * 1.6 * dt;
  pitch = Math.max(-1.55, Math.min(1.55, pitch + (k('ArrowUp') - k('ArrowDown')) * 1.0 * dt));
  right.set(Math.cos(yaw), 0, -Math.sin(yaw));
  if (mode === 'fly') {
    if (level) fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));          // heading only: altitude kept
    else fwd.set(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
    mv.set(0, 0, 0).addScaledVector(fwd, k('KeyW') - k('KeyS') + joy.f).addScaledVector(right, k('KeyD') - k('KeyA') + joy.r);
    mv.y += Math.min(1, k('Space') + k('KeyE')) - Math.min(1, k('Period') + k('KeyC') + k('KeyQ')) + joy.u;
    if (mv.lengthSq() > 1) mv.normalize();                          // keys: full speed; touch stick: proportional
    // phone: speed shrinks near the ground (below ~800 m) so one can come down gently and stand at eye height
    const g0 = MOBILE ? groundAt(camera.position.x, camera.position.z) : null;
    const near = g0 === null ? 1 : Math.max(0.006, Math.min(1, (camera.position.y - g0 + 5) / 800));
    camera.position.addScaledVector(mv, flySpeed * boost * near * dt);
    const g = groundAt(camera.position.x, camera.position.z), eye = MOBILE ? 1.8 : 3;
    if (g !== null && camera.position.y < g + eye) camera.position.y = g + eye;
  } else {
    fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    mv.set(0, 0, 0).addScaledVector(fwd, k('KeyW') - k('KeyS') + joy.f).addScaledVector(right, k('KeyD') - k('KeyA') + joy.r);
    if (mv.lengthSq() > 1) mv.normalize();
    camera.position.addScaledVector(mv, walkSpeed * (boost > 1 ? 5 : 1) * dt);
    const g = groundAt(camera.position.x, camera.position.z);
    if (g !== null) {
      vy -= 9.81 * dt; camera.position.y += vy * dt;
      if (camera.position.y <= g + 1.8) { camera.position.y = g + 1.8; vy = keys.has('Space') ? 5 : 0; }
    }
  }
  camera.rotation.set(pitch, yaw, 0);
  camera.updateMatrixWorld();
  applyQuality(dt);
  simTime += dt * 1000;
  setSun(latOfZ(camera.position.z), lonOfX(camera.position.x));
  sky.position.copy(camera.position);
  updateLOD(frame);
  updateSun(frame);
  updateSat(); updateFeat(); updateTrails(frame); updateTunnels(); update3d(); light3d();
  renderer.render(scene, camera);
  updateLabels(frame);

  measSpeed = 0.9 * measSpeed + 0.1 * prev.distanceTo(camera.position) / Math.max(dt, 1e-3);
  fpsN++; fpsT += dt; hudT += dt; hashT += dt;
  if (fpsT > 0.5) { fps = fpsN / fpsT; fpsN = 0; fpsT = 0; }
  if (hudT > 0.15) {
    hudT = 0;
    const P = camera.position, lat = latOfZ(P.z), lon = lonOfX(P.x), g = groundAt(P.x, P.z);
    const hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
    $('h-pos').textContent = `${lat.toFixed(4)}° N  ${lon.toFixed(4)}° E`;
    $('h-alt').textContent = `${P.y.toFixed(0)} m`;
    $('h-agl').textContent = g === null ? '–' : `${(P.y - g).toFixed(0)} m`;
    $('h-gnd').textContent = g === null ? '–' : `${g.toFixed(0)} m`;
    $('h-hdg').textContent = `${hdg.toFixed(0)}°  pitch ${(pitch * 180 / Math.PI).toFixed(0)}°`;
    $('h-spd').textContent = `${(measSpeed * 3.6).toFixed(0)} km/h`;
    $('h-set').textContent = speedText().replace(/ · .*/, '');
    $('h-lod').textContent = `${drawn.length} / ${chunks.size}  (+${inflight})`;
    $('h-fps').textContent = fps.toFixed(0);
    $('h-time').textContent = fmtTime(simTime);
    $('h-sunpos').textContent = `az ${sunAz.toFixed(1)}°  el ${sunEl.toFixed(1)}°`;
    if (shared.sunMode.value > 0) {
      const sv = sunAt(P.x, P.z);
      $('h-sun').textContent = sv === null ? '…' : sv === 'sea' ? 'sea' : `${Math.round(sv[0] / 10) * 10} h/yr, ${sv[1]} sunless d`;
    }
    drawMini();
  }
  if (hashT > 1.0) {
    hashT = 0;
    const P = camera.position, hdg = ((-yaw * 180 / Math.PI) % 360 + 360) % 360;
    history.replaceState(null, '', `#${latOfZ(P.z).toFixed(5)},${lonOfX(P.x).toFixed(5)},${P.y.toFixed(0)},${hdg.toFixed(1)},${(pitch * 180 / Math.PI).toFixed(1)}`);
  }
}
tick();
