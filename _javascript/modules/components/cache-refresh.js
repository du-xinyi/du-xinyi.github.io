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

  const registration = await navigator.serviceWorker.getRegistration();
  if (!registration) {
    return false;
  }

  await registration.update();

  const installingWorker = registration.installing;
  if (installingWorker) {
    await new Promise((resolve) => {
      const onStateChange = () => {
        if (
          installingWorker.state === 'installed' ||
          installingWorker.state === 'activated' ||
          installingWorker.state === 'redundant'
        ) {
          installingWorker.removeEventListener('statechange', onStateChange);
          resolve();
        }
      };

      installingWorker.addEventListener('statechange', onStateChange);
      onStateChange();
    });
  }

  const waitingWorker = registration.waiting;

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
    if (button.getAttribute('aria-disabled') === 'true') {
      return;
    }

    button.setAttribute('aria-disabled', 'true');
    button.classList.add('refreshing');

    try {
      const activatingUpdate = await updateServiceWorker();

      if (activatingUpdate) {
        // app.js reloads once the updated worker actually takes control.
        return;
      }
    } catch {
      // Reloading with a unique URL still bypasses a stale document response.
    }

    reloadWithCacheBuster();
  });
}
