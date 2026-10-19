importScripts('./assets/js/data/swconf.js');

const purge = swconf.purge;
const interceptor = swconf.interceptor;

function verifyUrl(url) {
  const requestUrl = new URL(url);
  const requestPath = requestUrl.pathname;

  if (!requestUrl.protocol.startsWith('http')) {
    return false;
  }

  for (const prefix of interceptor.urlPrefixes) {
    if (requestUrl.href.startsWith(prefix)) {
      return false;
    }
  }

  for (const path of interceptor.paths) {
    if (requestPath.startsWith(path)) {
      return false;
    }
  }
  return true;
}

function shouldRefreshFromNetwork(request) {
  const url = new URL(request.url);

  return (
    url.origin === self.location.origin ||
    request.mode === 'navigate' ||
    url.searchParams.has('v') ||
    url.searchParams.has('refresh')
  );
}

function cacheKey(request) {
  if (request.mode !== 'navigate') {
    return request;
  }

  const url = new URL(request.url);
  url.searchParams.delete('refresh');
  return url.href;
}

async function cacheResponse(request, response) {
  if (
    purge ||
    request.method !== 'GET' ||
    !response ||
    !response.ok ||
    !verifyUrl(request.url)
  ) {
    return;
  }

  try {
    const cache = await caches.open(swconf.cacheName);
    await cache.put(cacheKey(request), response.clone());
  } catch (error) {
    // A full cache must not discard a successful network response.
    console.warn('Unable to cache response:', error);
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request, { cache: 'no-cache' });
    await cacheResponse(request, response);
    return response;
  } catch {
    return caches.match(cacheKey(request));
  }
}

async function cacheFirst(request) {
  const cachedResponse = await caches.match(request);

  if (cachedResponse) {
    return cachedResponse;
  }

  const response = await fetch(request);
  await cacheResponse(request, response);
  return response;
}

self.addEventListener('install', (event) => {
  if (purge) {
    return;
  }

  event.waitUntil(
    caches.open(swconf.cacheName).then((cache) => {
      return cache.addAll(
        swconf.resources.map((url) => new Request(url, { cache: 'reload' }))
      );
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keyList) => {
        return Promise.all(
          keyList.map((key) => {
            if (purge) {
              return caches.delete(key);
            } else {
              if (key !== swconf.cacheName) {
                return caches.delete(key);
              }
            }
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (event) => {
  if (event.data === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener('fetch', (event) => {
  if (
    purge ||
    event.request.method !== 'GET' ||
    event.request.headers.has('range') ||
    !verifyUrl(event.request.url)
  ) {
    return;
  }

  event.respondWith(
    shouldRefreshFromNetwork(event.request)
      ? networkFirst(event.request)
      : cacheFirst(event.request)
  );
});
