'use strict';

const state = {
  payload: null,
  tab: 'ANIME',
  sortBy: 'rec_score',
  format: '',
  yearMin: null,
  yearMax: null,
  search: '',
  genres: new Set(),
};

function loadPayload() {
  const inline = document.getElementById('predictions-data');
  if (inline && inline.textContent && inline.textContent.trim() && inline.textContent.trim() !== '{}') {
    try {
      const data = JSON.parse(inline.textContent);
      if (data && Array.isArray(data.items)) return Promise.resolve(data);
    } catch (e) {
      console.warn('failed to parse inline predictions', e);
    }
  }
  return fetch('predictions.json', { cache: 'no-store' })
    .then((r) => {
      if (!r.ok) throw new Error('predictions.json HTTP ' + r.status);
      return r.json();
    });
}

function fmtScore(v) {
  if (v == null || isNaN(v)) return '—';
  return Number(v).toFixed(1);
}

function fmtRec(v) {
  if (v == null || isNaN(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return sign + Number(v).toFixed(1);
}

function buildHeader(payload) {
  const meta = payload.metadata || {};
  const tagline = document.getElementById('meta-tagline');
  const stats = document.getElementById('meta-stats');
  const trained = meta.trained_at ? new Date(meta.trained_at).toLocaleString() : 'unknown';
  tagline.textContent = `Personal anime + manga predictions for ${meta.user || '—'}`;
  stats.innerHTML = '';
  const cells = [
    ['Model', meta.model_name || '—'],
    ['Test MAE', meta.test_mae != null ? meta.test_mae.toFixed(2) : '—'],
    ['Test R²', meta.test_r2 != null ? meta.test_r2.toFixed(2) : '—'],
    ['Items', meta.n_items != null ? meta.n_items : (payload.items || []).length],
    ['Trained', trained],
  ];
  for (const [k, v] of cells) {
    const div = document.createElement('div');
    div.className = 'stat';
    div.innerHTML = `${k}: <strong>${v}</strong>`;
    stats.appendChild(div);
  }
}

function populateFilters(items) {
  const formats = new Set();
  const genres = new Set();
  for (const it of items) {
    if (it.format) formats.add(it.format);
    for (const g of it.genres || []) genres.add(g);
  }
  const fmtSel = document.getElementById('filter-format');
  for (const f of [...formats].sort()) {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    fmtSel.appendChild(opt);
  }

  const wrap = document.getElementById('genre-chips');
  for (const g of [...genres].sort()) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = g;
    chip.addEventListener('click', () => {
      if (state.genres.has(g)) state.genres.delete(g);
      else state.genres.add(g);
      chip.classList.toggle('active');
      render();
    });
    wrap.appendChild(chip);
  }
}

function applyFilters(items) {
  return items.filter((it) => {
    if (it.type !== state.tab) return false;
    if (state.format && it.format !== state.format) return false;
    if (state.yearMin != null && (it.year == null || it.year < state.yearMin)) return false;
    if (state.yearMax != null && (it.year == null || it.year > state.yearMax)) return false;
    if (state.genres.size > 0) {
      const set = new Set(it.genres || []);
      for (const g of state.genres) if (!set.has(g)) return false;
    }
    if (state.search) {
      const needle = state.search.toLowerCase();
      const hay = [it.title_english, it.title_romaji, it.title_native].filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });
}

function sortItems(items) {
  const key = state.sortBy;
  return items.slice().sort((a, b) => {
    const av = a[key] == null ? -Infinity : a[key];
    const bv = b[key] == null ? -Infinity : b[key];
    return bv - av;
  });
}

function render() {
  const items = state.payload.items || [];
  const filtered = applyFilters(items);
  const sorted = sortItems(filtered);

  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  if (!sorted.length) {
    const div = document.createElement('div');
    div.className = 'empty';
    div.textContent = 'No items match the current filters.';
    grid.appendChild(div);
    return;
  }
  const frag = document.createDocumentFragment();
  for (const it of sorted) frag.appendChild(buildCard(it));
  grid.appendChild(frag);
}

function buildCard(it) {
  const card = document.createElement('a');
  card.className = 'card';
  card.href = it.site_url || '#';
  card.target = '_blank';
  card.rel = 'noopener noreferrer';

  const cover = document.createElement('div');
  cover.className = 'card-cover';
  if (it.cover_image) {
    const img = document.createElement('img');
    img.src = it.cover_image;
    img.alt = it.title_english || it.title_romaji || '';
    img.loading = 'lazy';
    cover.appendChild(img);
  }
  const badge = document.createElement('div');
  badge.className = 'rec-badge';
  badge.textContent = fmtRec(it.rec_score);
  cover.appendChild(badge);

  const body = document.createElement('div');
  body.className = 'card-body';

  const title = document.createElement('div');
  title.className = 'card-title';
  title.textContent = it.title_english || it.title_romaji || '(untitled)';
  body.appendChild(title);

  const sub = document.createElement('div');
  sub.className = 'card-sub';
  const subParts = [];
  if (it.year) subParts.push(it.year);
  if (it.format) subParts.push(it.format);
  if (it.episodes) subParts.push(`${it.episodes} ep`);
  else if (it.chapters) subParts.push(`${it.chapters} ch`);
  if (it.source && it.source !== 'ORIGINAL') subParts.push(it.source.toLowerCase().replace(/_/g, ' '));
  sub.textContent = subParts.join(' · ');
  body.appendChild(sub);

  if (it.genres && it.genres.length) {
    const gwrap = document.createElement('div');
    gwrap.className = 'card-genres';
    for (const g of it.genres.slice(0, 4)) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = g;
      gwrap.appendChild(chip);
    }
    body.appendChild(gwrap);
  }

  const scores = document.createElement('div');
  scores.className = 'card-scores';
  scores.appendChild(scoreCell('Predicted', fmtScore(it.predicted_score)));
  scores.appendChild(scoreCell('Mean', fmtScore(it.mean_score)));
  const recCell = scoreCell('Rec', fmtRec(it.rec_score), 'rec');
  if (it.rec_score != null && it.rec_score < 0) recCell.classList.add('neg');
  scores.appendChild(recCell);
  body.appendChild(scores);

  card.appendChild(cover);
  card.appendChild(body);
  return card;
}

function scoreCell(label, value, extra) {
  const cell = document.createElement('div');
  cell.className = 'score-cell' + (extra ? ' ' + extra : '');
  cell.innerHTML = `<span class="label">${label}</span><span class="value">${value}</span>`;
  return cell;
}

function updateTabCounts() {
  const items = state.payload.items || [];
  let a = 0, m = 0;
  for (const it of items) {
    if (it.type === 'ANIME') a++;
    else if (it.type === 'MANGA') m++;
  }
  document.getElementById('count-anime').textContent = a;
  document.getElementById('count-manga').textContent = m;
}

function bindControls() {
  for (const btn of document.querySelectorAll('.tab')) {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.tab = btn.dataset.tab;
      render();
    });
  }
  document.getElementById('sort-by').addEventListener('change', (e) => {
    state.sortBy = e.target.value;
    render();
  });
  document.getElementById('filter-format').addEventListener('change', (e) => {
    state.format = e.target.value;
    render();
  });
  document.getElementById('year-min').addEventListener('input', (e) => {
    const v = parseInt(e.target.value, 10);
    state.yearMin = isNaN(v) ? null : v;
    render();
  });
  document.getElementById('year-max').addEventListener('input', (e) => {
    const v = parseInt(e.target.value, 10);
    state.yearMax = isNaN(v) ? null : v;
    render();
  });
  document.getElementById('search').addEventListener('input', (e) => {
    state.search = e.target.value.trim();
    render();
  });
  document.getElementById('genre-clear').addEventListener('click', () => {
    state.genres.clear();
    document.querySelectorAll('#genre-chips .chip').forEach((c) => c.classList.remove('active'));
    render();
  });
}

loadPayload()
  .then((payload) => {
    state.payload = payload;
    buildHeader(payload);
    populateFilters(payload.items || []);
    updateTabCounts();
    bindControls();
    render();
  })
  .catch((err) => {
    console.error(err);
    const grid = document.getElementById('grid');
    grid.innerHTML = `<div class="empty">Failed to load predictions.json — ${err.message}</div>`;
  });
