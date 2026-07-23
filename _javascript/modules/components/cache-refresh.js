const REFRESH_PARAM = 'refresh';

function cleanRefreshParam() {
  const url = new URL(window.location.href);

  if (!url.searchParams.has(REFRESH_PARAM)) {
    return;
  }

  url.searchParams.delete(REFRESH_PARAM);
  window.history.replaceState(null, '', url);
}

async function clearSiteCache() {
  if ('serviceWorker' in navigator) {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((registration) => registration.unregister()));
  }

  if ('caches' in window) {
    const cacheNames = await window.caches.keys();
    await Promise.all(cacheNames.map((name) => window.caches.delete(name)));
  }
}

export function initCacheRefresh() {
  cleanRefreshParam();

  const button = document.getElementById('cache-refresh');

  if (!button) {
    return;
  }

  button.addEventListener('click', async () => {
    button.disabled = true;
    button.classList.add('refreshing');

    try {
      await clearSiteCache();
    } finally {
      const url = new URL(window.location.href);
      url.searchParams.set(REFRESH_PARAM, Date.now().toString());
      window.location.replace(url);
    }
  });
}
