// Alps Flight offline cache (build_mobile.py web). App shell: network first (updates), cached copy offline.
// Terrain / path tiles: cache first, stored the first time they are seen.
const V = '202611221749', SHELL = ["./", "index.html", "flight.js", "manifest.webmanifest", "icon-192.png", "tiles/meta.json", "tiles/overview.png", "labels.json", "water.json", "features.json", "trails/tunnels/tunnels.json", "lib/three.module.js", "lib/addons/lines/LineSegments2.js", "lib/addons/lines/LineSegmentsGeometry.js", "lib/addons/lines/LineMaterial.js", "trails/osm/index.json", "trails/roads/index.json", "trails/ch/index.json"];
self.addEventListener('install', e => e.waitUntil(caches.open('shell-' + V).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(
  ks.filter(k => k.startsWith('shell-') && k !== 'shell-' + V).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return;
  const tile = /\/(tiles|trails)\/.+\.(webp|bin|gz)$/.test(u.pathname);
  if (tile) {
    e.respondWith(caches.open('tiles').then(c => c.match(e.request).then(r => r || fetch(e.request).then(resp => {
      if (resp.ok) c.put(e.request, resp.clone());
      return resp;
    }))));
  } else {
    e.respondWith(fetch(e.request).then(resp => {
      if (resp.ok) { const cl = resp.clone(); caches.open('shell-' + V).then(c => c.put(e.request, cl)); }
      return resp;
    }).catch(() => caches.match(e.request, { ignoreSearch: true })));
  }
});
