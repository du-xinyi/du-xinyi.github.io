/**
 * Sets up the mode toggle dropdown, allowing users to switch between light, dark, and system themes.
 *
 * Dependencies:
 *  - Theme (${JS_ROOT}/theme.js)
 */

const ACTIVE_CLASS = 'active';
const SHOW_CLASS = 'show';
const toggle = document.getElementById('mode-toggle');
const dropdown = document.querySelector('#mode-toggle + .dropdown-menu');
const activeMode = Theme.isSystemTheme
  ? Theme.Mode.SYSTEM
  : Theme.resolvedTheme;

function closeDropdown() {
  dropdown.classList.remove(SHOW_CLASS);
  toggle.setAttribute('aria-expanded', 'false');
}

export function modeWatcher() {
  if (!Theme.isToggleable) {
    return;
  }

  toggle.addEventListener('click', (event) => {
    event.stopPropagation();
    const shouldOpen = !dropdown.classList.contains(SHOW_CLASS);
    dropdown.classList.toggle(SHOW_CLASS, shouldOpen);
    toggle.setAttribute('aria-expanded', String(shouldOpen));
  });

  document.addEventListener('click', (event) => {
    if (!event.target.closest('.sidebar-theme-actions')) {
      closeDropdown();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeDropdown();
      toggle.focus();
    }
  });

  dropdown
    .querySelectorAll('.dropdown-item[data-theme-mode]')
    .forEach((option) => {
    const mode = option.dataset.themeMode;
    if (mode === activeMode) {
      option.classList.add(ACTIVE_CLASS);
        return;
      }
    });

  dropdown.addEventListener('click', (event) => {
    const current = event.target.closest('.dropdown-item[data-theme-mode]');

    if (!current) {
      return;
    }

    const lastActive = dropdown.querySelector(`.${ACTIVE_CLASS}`);

    if (lastActive === current) {
      closeDropdown();
      return;
    }

    lastActive?.classList.remove(ACTIVE_CLASS);
    current.classList.add(ACTIVE_CLASS);
    Theme.update(current.dataset.themeMode);
    closeDropdown();
  });
}
