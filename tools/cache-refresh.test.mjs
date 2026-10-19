import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const origin = 'https://du-xinyi.github.io';
const source = (path) => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');

function serviceWorker({ offline = false, cacheFull = false, purge = false } = {}) {
  const entries = new Map();
  const fetches = [];
  const handlers = {};
  const key = (request) => typeof request === 'string' ? request : request.url;
  const cache = {
    async put(request, response) {
      if (cacheFull) throw new Error('QuotaExceededError');
      entries.set(key(request), response);
    },
    async addAll(requests) {
      for (const request of requests) {
        assert.equal(request.cache, 'reload');
        entries.set(request.url, new Response('precache'));
      }
    }
  };
  let claimed = false;
  const context = vm.createContext({
    URL,
    Request: class extends Request {
      constructor(url, options) { super(new URL(url, origin), options); }
    },
    console: { warn() {} },
    importScripts() {},
    swconf: {
      purge,
      cacheName: 'current',
      resources: ['/', '/assets/js/data/search.json?v=123'],
      interceptor: { paths: ['/excluded/'], urlPrefixes: [] }
    },
    self: {
      location: { origin },
      clients: { async claim() { claimed = true; } },
      addEventListener(type, handler) { handlers[type] = handler; }
    },
    caches: {
      async open() { return cache; },
      async match(request) { return entries.get(key(request))?.clone(); },
      async keys() { return ['current']; }
    },
    async fetch(request, options) {
      fetches.push({ request, options });
      if (offline) throw new TypeError('Offline');
      return new Response('new content');
    }
  });
  vm.runInContext(source('_javascript/pwa/sw.js'), context);
  return {
    entries,
    fetches,
    get claimed() { return claimed; },
    async lifecycle(type) {
      let task;
      handlers[type]({ waitUntil(promise) { task = promise; } });
      await task;
    },
    async request(path, options = {}) {
      let task;
      handlers.fetch({
        request: {
          url: origin + path,
          mode: 'cors',
          method: 'GET',
          headers: new Headers(),
          ...options
        },
        respondWith(promise) { task = promise; }
      });
      return task;
    }
  };
}

test('刷新页面后，搜索索引、脚本和图片均获取新内容并重新验证 HTTP 缓存', async () => {
  const sw = serviceWorker();
  for (const path of ['/?refresh=123', '/assets/js/data/search.json',
    '/assets/js/collapsible-headings.js', '/assets/img/avatar.png']) {
    sw.entries.set(origin + path, new Response('old content'));
    const response = await sw.request(path, { mode: path.startsWith('/?') ? 'navigate' : 'cors' });
    assert.equal(await response.text(), 'new content');
  }
  assert.equal(sw.fetches.length, 4);
  assert.ok(sw.fetches.every(({ options }) => options.cache === 'no-cache'));
});

test('离线时仍可读取缓存，刷新参数不破坏页面回退', async () => {
  const sw = serviceWorker({ offline: true });
  sw.entries.set(origin + '/', new Response('offline home'));
  sw.entries.set(origin + '/assets/js/data/search.json?v=123', new Response('offline search'));
  assert.equal(await (await sw.request('/?refresh=123', { mode: 'navigate' })).text(), 'offline home');
  assert.equal(await (await sw.request('/assets/js/data/search.json?v=123')).text(), 'offline search');
});

test('缓存写入失败不会用旧内容覆盖成功的网络响应', async () => {
  const sw = serviceWorker({ cacheFull: true });
  sw.entries.set(origin + '/', new Response('old content'));
  assert.equal(await (await sw.request('/', { mode: 'navigate' })).text(), 'new content');
});

test('安装时重新获取预缓存资源，激活时接管页面', async () => {
  const sw = serviceWorker();
  await sw.lifecycle('install');
  assert.equal(sw.entries.size, 2);
  assert.equal(sw.claimed, false);
  await sw.lifecycle('activate');
  assert.equal(sw.claimed, true);
});

test('禁用缓存、排除路径、POST 和 Range 请求不被离线缓存拦截', async () => {
  assert.equal(await serviceWorker({ purge: true }).request('/'), undefined);
  const sw = serviceWorker();
  assert.equal(await sw.request('/excluded/page'), undefined);
  assert.equal(await sw.request('/', { method: 'POST' }), undefined);
  assert.equal(await sw.request('/', { headers: new Headers({ range: 'bytes=0-10' }) }), undefined);
  assert.equal(sw.fetches.length, 0);
});

test('慢速安装完成后激活新版，并仅在接管页面后重载一次', async () => {
  const installing = new EventTarget();
  installing.state = 'installing';
  const serviceWorkers = new EventTarget();
  const registration = new EventTarget();
  registration.installing = installing;
  registration.waiting = null;
  let checked;
  const updateChecked = new Promise((resolve) => { checked = resolve; });
  registration.update = async () => { checked(); };
  serviceWorkers.getRegistration = async () => registration;
  serviceWorkers.register = async (url, options) => {
    assert.equal(url, '/sw.min.js');
    assert.equal(options.updateViaCache, 'none');
    return registration;
  };
  let click;
  let disabled = null;
  const reloads = [];
  const button = {
    addEventListener(type, handler) { click = handler; },
    getAttribute() { return disabled; },
    setAttribute(name, value) { disabled = value; },
    classList: { add() {} }
  };
  const context = vm.createContext({
    URL, Date,
    navigator: { serviceWorker: serviceWorkers },
    document: {
      currentScript: { src: origin + '/app.min.js?baseurl=&register=true&v=123' },
      getElementById(id) {
        return id === 'cache-refresh' ? button : { querySelector() { return { addEventListener() {} }; } };
      }
    },
    Toast: { getOrCreateInstance() { return { show() {}, hide() {} }; } },
    window: {
      location: { href: origin + '/?query=keep#section', replace(url) { reloads.push(new URL(url)); } },
      setTimeout() { throw new Error('必须等待状态事件，不能用固定延时触发刷新'); }
    }
  });
  vm.runInContext(source('_javascript/pwa/app.js').replace(/^import .*;\n/, ''), context);
  vm.runInContext(source('_javascript/modules/components/cache-refresh.js').replace('export function', 'function'), context);
  vm.runInContext('initCacheRefresh()', context);
  const refreshing = click({ preventDefault() {} });
  await updateChecked;
  // Simulate installation remaining in progress beyond the former three-second limit.
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(reloads.length, 0);
  let messages = 0;
  registration.waiting = {
    postMessage(message) { assert.equal(message, 'SKIP_WAITING'); messages++; }
  };
  registration.installing = null;
  installing.state = 'installed';
  installing.dispatchEvent(new Event('statechange'));
  await refreshing;
  assert.equal(messages, 1);
  assert.equal(reloads.length, 0);
  serviceWorkers.dispatchEvent(new Event('controllerchange'));
  serviceWorkers.dispatchEvent(new Event('controllerchange'));
  assert.equal(reloads.length, 1);
  assert.ok(reloads[0].searchParams.has('refresh'));
  assert.equal(reloads[0].searchParams.get('query'), 'keep');
  assert.equal(reloads[0].hash, '#section');
});
