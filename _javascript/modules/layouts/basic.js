import {
  back2top,
  initCacheRefresh,
  loadTooltip,
  modeWatcher
} from '../components';

export function basic() {
  modeWatcher();
  initCacheRefresh();
  back2top();
  loadTooltip();
}
