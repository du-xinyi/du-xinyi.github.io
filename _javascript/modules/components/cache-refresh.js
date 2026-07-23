const REFRESH_PARAM = 'refresh';

function cleanRefreshParam() {
  const url = new URL(window.location.href);

  if (!url.searchParams.has(REFRESH_PARAM)) {
    return;
  }

  url.searchParams.delete(REFRESH_PARAM);
  window.history.replaceState(null, '', url);
}

async function updateServiceWorker() {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  const registrations = await navigator.serviceWorker.getRegistrations();
  await Promise.all(registrations.map((registration) => registration.update()));

  const waitingWorker = registrations
    .map((registration) => registration.waiting)
    .find(Boolean);

  if (!waitingWorker) {
    return false;
  }

  waitingWorker.postMessage('SKIP_WAITING');
  return true;
}

function reloadWithCacheBuster() {
  const url = new URL(window.location.href);
  url.searchParams.set(REFRESH_PARAM, Date.now().toString());
  window.location.replace(url);
}

export function initCacheRefresh() {
  cleanRefreshParam();

  const button = document.getElementById('cache-refresh');

  if (!button) {
    return;
  }

  button.addEventListener('click', async (event) => {
    event.preventDefault();
    button.setAttribute('aria-disabled', 'true');
    button.classList.add('refreshing');

    try {
      const activatingUpdate = await updateServiceWorker();

      if (activatingUpdate) {
        window.setTimeout(reloadWithCacheBuster, 1500);
        return;
      }
    } catch {
      // Reloading with a unique URL still bypasses a stale document response.
    }

    reloadWithCacheBuster();
  });
}
