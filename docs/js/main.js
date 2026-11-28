/* Into Frame site: the navbar's menu on narrow screens, and the sidebar's
   current section highlighted while scrolling. */
'use strict';

const toggle = document.querySelector('.navbar__toggle');
const links = document.getElementById('navbar-links');

if (toggle && links) {
  const close = () => { links.classList.remove('is-open'); toggle.setAttribute('aria-expanded', 'false'); };
  toggle.addEventListener('click', () => {
    const open = links.classList.toggle('is-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  links.querySelectorAll('a').forEach((a) => a.addEventListener('click', close));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
}

const heads = document.querySelectorAll('main h2[id]');
const sideLinks = document.querySelectorAll('.sidebar a[href^="#"]');

if (heads.length && sideLinks.length && 'IntersectionObserver' in window) {
  const mark = (id) => sideLinks.forEach((a) => a.classList.toggle('is-active', a.getAttribute('href') === `#${id}`));
  const seen = new Map();
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => seen.set(e.target.id, e.isIntersecting));
    // The first heading on screen, or the last one scrolled past.
    const onScreen = [...heads].find((h) => seen.get(h.id));
    if (onScreen) { mark(onScreen.id); return; }
    const past = [...heads].filter((h) => h.getBoundingClientRect().top < 0).pop();
    if (past) mark(past.id);
  }, { rootMargin: '0px 0px -60% 0px' });
  heads.forEach((h) => io.observe(h));
}
