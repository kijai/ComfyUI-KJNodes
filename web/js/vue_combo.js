const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

// Nodes 2.0 combo widget tweaks: taller dropdown list, legacy-style prev/next arrows, folder tree view
const TRIGGER = '[data-testid="widget-select-default-trigger"]';
const VIEWPORT = '[data-testid="widget-select-default-viewport"]';
const OVERLAY = '[data-testid="widget-select-default-overlay"]';
// trigger button + chevron button, the two elements reka opens the dropdown from (on click)
const OPENERS = `${TRIGGER}, button[aria-hidden="true"]:has(> i[class*="chevron-down"])`;
const ARROW_CLASS = "kj-combo-arrow";
const SEP = /[\\/]/;
// "folder/file.ext"; keeps option text like "N/A" or "owner/model" on the flat list
const PATH_FILE = /[\\/][^\\/]*\.[a-z][a-z0-9]{0,15}$/i;
// root-level entries pinned above the folders, like rgthree's special entries
const SPECIAL = /^(none|disabled?|default|choose|open)(\s|$)/i;
const MOVES = { ArrowDown: 1, ArrowUp: -1, PageDown: 10, PageUp: -10 };
// before frontend 1.57.1 (virtualized combo, FE-1166) the native list mounts every option: 0.4s for 800, 24s for 20k
const LONG_LIST = 200;
const nativeListIsSlow = () => {
	const [major = 0, minor = 0, patch = 0] = String(window.__COMFYUI_FRONTEND_VERSION__ ?? "0").split(".").map((n) => parseInt(n) || 0);
	return major * 1e6 + minor * 1e3 + patch < 1057001;
};

const ICONS = {
	prev: '<path d="M15 18l-6-6 6-6"/>',
	next: '<path d="M9 18l6-6-6-6"/>',
	search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
	folder: '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
	check: '<path d="M20 6 9 17l-5-5"/>',
	star: '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
	clock: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
};
const svg = (name, size = 14, cls = "") =>
	`<svg class="${cls}" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
const h = (tag, props) => Object.assign(document.createElement(tag), props);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const pathKey = (s) => String(s).toLowerCase().replaceAll("\\", "/");
const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const rootRem = () => parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
const docEl = document.documentElement;
const ago = (t) => {
	const s = Date.now() / 1000 - t;
	for (const [unit, n] of [["y", 31536000], ["mo", 2592000], ["w", 604800], ["d", 86400], ["h", 3600], ["m", 60]]) if (s >= n) return `${Math.floor(s / n)}${unit}`;
	return "now";
};

// favorites, recent picks and sort order; global, each list shows only the entries it contains
const PICKS_KEY = "KJNodes.vueCombo.picks";
const picks = { recent: [], fav: [], sort: "name" };
try {
	Object.assign(picks, JSON.parse(localStorage.getItem(PICKS_KEY)));
} catch {}
const savePicks = () => {
	try {
		localStorage.setItem(PICKS_KEY, JSON.stringify(picks));
	} catch {}
};

const CSS = `
.${ARROW_CLASS} {
	display: flex; flex-shrink: 0; align-items: center; justify-content: center;
	width: 1.25rem; height: 100%; padding: 0; border: none; outline: none;
	background: transparent; color: inherit; opacity: 0.6; cursor: pointer;
}
/* one step past the select's own hover color, so the arrow stands out while the whole box is hovered */
.${ARROW_CLASS}:hover { opacity: 1; background: var(--component-node-widget-background-selected, var(--secondary-background-selected)); }
div:has(> ${TRIGGER}:disabled) > .${ARROW_CLASS} { pointer-events: none; }
/* rows sit at a 2rem pitch after 0.25rem padding (virtualized list, frontend 1.57+); placement-independent cap so reka's flip can work */
:root[data-kj-combo-rows] ${VIEWPORT} {
	max-height: min(calc(var(--kj-combo-rows) * 2rem + 0.25rem), calc(var(--kj-combo-space, var(--reka-combobox-content-available-height, 100vh)) - 3rem)) !important;
}
/* list taller than either side of the trigger: pin reka's popper to the viewport bottom, covering the widget */
:root[data-kj-combo-pin] [data-reka-popper-content-wrapper]:has(${OVERLAY}) {
	top: auto !important; bottom: 8px !important;
	transform: translateX(min(var(--kj-combo-x), calc(100vw - 100% - 8px))) !important;
}
.kj-ctree {
	position: fixed; z-index: 3000; display: flex; flex-direction: column; overflow: hidden;
	background: var(--base-background); color: var(--base-foreground); font-size: 0.75rem;
	border: 1px solid var(--border-default); border-radius: 0.5rem; box-shadow: 0 4px 16px rgb(0 0 0 / 0.3);
}
.kj-ctree-search {
	display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem; padding: 0.375rem 0.5rem;
	border: 1px solid var(--border-default); border-radius: 0.375rem; color: var(--muted-foreground);
}
.kj-ctree-search:focus-within { border-color: var(--primary-background); }
.kj-ctree-search input {
	flex: 1; min-width: 0; padding: 0; border: none; outline: none;
	background: transparent; color: var(--base-foreground); font: inherit;
}
.kj-ctree-sort { display: flex; padding: 0; border: none; background: none; color: inherit; cursor: pointer; opacity: 0.6; }
.kj-ctree-sort:hover, .kj-ctree-sort.on { opacity: 1; color: var(--primary-background); }
/* rows and headers share one fixed height so the list can be virtualized */
.kj-ctree-head {
	display: flex; align-items: flex-end; height: 1.75rem; padding: 0 0.5rem 0.25rem; box-sizing: border-box;
	font-size: 0.625rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted-foreground);
}
.kj-ctree-list { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 0.25rem; scrollbar-width: thin; }
.kj-ctree-row {
	display: flex; align-items: center; gap: 0.25rem; height: 1.75rem; padding-right: 0.5rem;
	border-radius: 0.25rem; cursor: pointer; user-select: none; white-space: nowrap;
}
.kj-ctree-row.active { background: var(--secondary-background); }
.kj-ctree-row.selected { background: color-mix(in srgb, var(--primary-background) 20%, transparent); }
.kj-ctree-row.selected.active { background: color-mix(in srgb, var(--primary-background) 32%, transparent); }
.kj-ctree-row svg { flex-shrink: 0; color: var(--muted-foreground); }
.kj-ctree-folder.open .kj-ctree-chev { transform: rotate(90deg); }
.kj-ctree-sub { width: max-content; min-width: 12.5rem; max-width: 26rem; }
.kj-ctree-menus .kj-ctree-folder.open { background: var(--secondary-background); }
.kj-ctree-folder.holds .kj-ctree-name { color: var(--primary-background); }
.kj-ctree-name { flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.kj-ctree-count { margin-left: auto; padding-left: 0.75rem; color: var(--muted-foreground); }
/* path gives way before the name and clips from the left, keeping the nearest folders visible */
.kj-ctree-path {
	flex: 0 100 auto; min-width: 30%; margin-left: auto; padding-left: 0.75rem;
	overflow: hidden; text-overflow: ellipsis; direction: rtl; color: var(--muted-foreground);
}
.kj-ctree-row .kj-ctree-check { color: var(--base-foreground); }
.kj-ctree-count + svg { margin-left: 0.25rem; }
.kj-ctree-end { display: flex; flex-shrink: 0; align-items: center; gap: 0.25rem; margin-left: auto; padding-left: 0.5rem; }
.kj-ctree-path + .kj-ctree-end { margin-left: 0; }
.kj-ctree-age { color: var(--muted-foreground); }
.kj-ctree-star { display: flex; opacity: 0; }
.kj-ctree-row:hover .kj-ctree-star, .kj-ctree-row.active .kj-ctree-star { opacity: 0.6; }
.kj-ctree-star.on { opacity: 1; }
.kj-ctree-star.on svg { color: var(--primary-background); fill: currentColor; }
.kj-ctree-row mark { background: none; color: var(--primary-background); font-weight: 600; }
.kj-ctree-note { padding: 0.5rem; color: var(--muted-foreground); }
`;
document.getElementById("kj-vue-combo-style")?.remove();
document.head.append(h("style", { id: "kj-vue-combo-style", textContent: CSS }));

let maxVisible = 0, folderMode = "disabled", observer, closeTree, reopenGuard;

// below the trigger if it fits, else above, else pinned to the viewport bottom over the trigger
function placement(anchor, want) {
	const r = anchor.getBoundingClientRect();
	const below = innerHeight - r.bottom - 8, above = r.top - 8;
	return want <= below ? { r, side: "below", space: below } : want <= above ? { r, side: "above", space: above } : { r, side: "pin", space: innerHeight - 16 };
}

function placeNative(anchor, count) {
	// rows of 2rem + search box (only shown above 4 options); unknown count assumes the full setting
	const { r, side } = placement(anchor, (Math.min(maxVisible, count ?? maxVisible) * 2 + (count <= 4 ? 0.5 : 2.75)) * rootRem());
	docEl.toggleAttribute("data-kj-combo-pin", side === "pin");
	docEl.style.setProperty("--kj-combo-x", `${Math.max(r.left, 8)}px`);
	// below/above is left to reka's flip; vh keeps a pinned list in view across window resizes
	docEl.style.setProperty("--kj-combo-space", side === "pin" ? "calc(100vh - 16px)" : `${Math.max(r.top, innerHeight - r.bottom) - 16}px`);
	// lock the opening height once reka has rendered and placed it, so filtering doesn't move an upward/pinned list
	requestAnimationFrame(() => requestAnimationFrame(() => {
		const content = document.querySelector(OVERLAY);
		if (content) content.style.minHeight = `${content.offsetHeight}px`;
	}));
}

function resolveCombo(trigger) {
	const nodeEl = trigger?.closest("[data-node-id]");
	if (!nodeEl || trigger.disabled) return null;
	const node = app.canvas?.graph?.getNodeById(nodeEl.dataset.nodeId);
	const label = trigger.getAttribute("aria-label");
	// same-labelled combos (e.g. promoted inputs) are matched by DOM order
	const twins = [...nodeEl.querySelectorAll(TRIGGER)].filter((t) => t.getAttribute("aria-label") === label);
	const widget = node?.widgets?.filter((w) => w.type === "combo" && (w.label || w.name) === label)[twins.indexOf(trigger)];
	if (!widget) return null;
	let values = widget.options?.values;
	if (typeof values === "function") values = values(widget, node);
	// the native list drops null/undefined options too
	return Array.isArray(values) ? { node, widget, values: values.filter((v) => v != null) } : null;
}

function indexOfValue(values, value) {
	const i = values.indexOf(value);
	return i !== -1 ? i : values.findIndex((v) => String(v) === String(value));
}

function applyValue({ node, widget }, value) {
	if (value === undefined || value === widget.value) return;
	// setValue fires callback/onWidgetChanged like the legacy canvas; promoted subgraph widgets are store projections without it
	if (widget.setValue) {
		widget.setValue(value, { node, canvas: app.canvas });
	} else {
		// same sequence as the frontend's setNodeWidgetValue; onWidgetChanged clears the source's widget errors
		const old = widget.value;
		widget.value = value;
		widget.callback?.(value, app.canvas, node);
		node.onWidgetChanged?.(widget.name, value, old, widget);
	}
	const tracker = app.extensionManager?.workflow?.activeWorkflow?.changeTracker;
	(tracker?.captureCanvasState ?? tracker?.checkState)?.call(tracker);
}

function makeArrow(delta) {
	const btn = h("button", { type: "button", tabIndex: -1, className: ARROW_CLASS, innerHTML: svg(delta > 0 ? "next" : "prev") });
	btn.setAttribute("aria-label", delta > 0 ? "Next option" : "Previous option");
	// keep focus off the button, else Space (canvas pan) would re-click it
	btn.addEventListener("mousedown", (e) => e.preventDefault());
	btn.addEventListener("dblclick", (e) => e.stopPropagation());
	btn.addEventListener("click", (e) => {
		e.preventDefault();
		e.stopPropagation();
		const combo = resolveCombo(btn.parentElement?.querySelector(`:scope > ${TRIGGER}`));
		if (combo) applyValue(combo, combo.values[clamp(indexOfValue(combo.values, combo.widget.value) + delta, 0, combo.values.length - 1)]);
	});
	return btn;
}

function scan(root) {
	for (const trigger of root.matches(TRIGGER) ? [root] : root.querySelectorAll(TRIGGER)) {
		const anchor = trigger.parentElement;
		if (!anchor || !trigger.closest("[data-node-id]") || anchor.querySelector(`:scope > .${ARROW_CLASS}`)) continue;
		anchor.prepend(makeArrow(-1));
		anchor.append(makeArrow(1));
	}
}

function setArrows(enabled) {
	if (enabled && !observer) {
		// tree rows are rebuilt on every folder toggle and never hold a trigger
		observer = new MutationObserver((records) => {
			for (const r of records) for (const n of r.addedNodes) if (n.nodeType === Node.ELEMENT_NODE && !n.classList.contains("kj-ctree-row")) scan(n);
		});
		observer.observe(document.body, { childList: true, subtree: true });
		scan(document.body);
	} else if (!enabled && observer) {
		observer.disconnect();
		observer = null;
		document.querySelectorAll(`.${ARROW_CLASS}`).forEach((el) => el.remove());
	}
}

function buildTree(values) {
	const root = { path: "", folders: new Map(), files: [], all: [], folderCount: 0 };
	for (const value of values) {
		const parts = String(value).split(SEP).filter(Boolean);
		let dir = root;
		for (const part of parts.slice(0, -1)) {
			if (!dir.folders.has(part)) {
				dir.folders.set(part, { name: part, path: dir.path ? `${dir.path}/${part}` : part, parent: dir, folders: new Map(), files: [], count: 0, mtime: 0 });
				root.folderCount++;
			}
			dir = dir.folders.get(part);
		}
		const name = parts.at(-1) ?? String(value);
		const file = { value, name, lname: name.toLowerCase(), parent: dir, key: pathKey(value), mtime: 0 };
		dir.files.push(file);
		root.all.push(file);
		for (let d = dir; d.parent; d = d.parent) d.count++;
	}
	return root;
}

// stable sort, so name order stays until the modified times arrive
const newestFirst = (items, byDate) => (byDate ? [...items].sort((a, b) => b.mtime - a.mtime) : items);

function treeRows(dir, expanded, depth, byDate, out = []) {
	const pinned = depth ? [] : dir.files.filter((f) => SPECIAL.test(f.name));
	for (const file of pinned) out.push({ file, depth });
	for (const folder of newestFirst([...dir.folders.values()], byDate)) {
		out.push({ folder, depth });
		if (expanded.has(folder)) treeRows(folder, expanded, depth + 1, byDate, out);
	}
	for (const file of newestFirst(dir.files, byDate)) if (!pinned.includes(file)) out.push({ file, depth });
	return out;
}

function searchFiles(files, terms, byDate) {
	// name starting with the first term, then name containing all terms, then path-only matches
	const ranks = [[], [], []];
	for (const f of files) if (terms.every((t) => f.key.includes(t))) ranks[f.lname.startsWith(terms[0]) ? 0 : terms.every((t) => f.lname.includes(t)) ? 1 : 2].push(f);
	return ranks.flatMap((r) => newestFirst(r, byDate));
}

// tree style is one panel with inline folders; submenus style adds a panel per open folder level
function openFolderTree(anchor, combo) {
	closeTree?.();
	const submenus = folderMode === "submenus";
	const tree = buildTree(combo.values);
	const current = tree.all[indexOfValue(combo.values, combo.widget.value)];
	const holders = new Set();
	for (let d = current?.parent; d?.parent; d = d.parent) holders.add(d);
	const expanded = new Set(submenus ? [] : holders);
	const maxRows = maxVisible || 15;
	const panels = [];
	const ac = new AbortController();
	const rem = rootRem(), rowH = 1.75 * rem;
	let focus = 0, query = "", hoverTimer = 0, byDate = picks.sort === "date", datesLoaded = false;
	const byValue = new Map(tree.all.map((f) => [String(f.value), f]));

	const span = (cls, html) => `<span class="kj-ctree-${cls}">${html}</span>`;
	let hitRe = null;
	// escape, wrapping search-term matches in <mark>
	const mark = (text) => (hitRe ? text.split(hitRe).map((s, k) => (k % 2 ? `<mark>${esc(s)}</mark>` : esc(s))).join("") : esc(text));
	const rowHtml = (p, row, i) => {
		if ("head" in row) return `<div class="kj-ctree-head">${row.head}</div>`;
		const f = row.folder ?? row.file;
		hitRe = p.hit;
		let cls = i === p.active ? "kj-ctree-row active" : "kj-ctree-row", pad, html;
		if (row.folder) {
			cls += ` kj-ctree-folder${expanded.has(f) || panels[p.level + 1]?.dir === f ? " open" : ""}${holders.has(f) ? " holds" : ""}`;
			pad = submenus ? 0.5 : 0.25 + row.depth;
			html = (submenus ? "" : svg("next", 12, "kj-ctree-chev")) + svg("folder") + span("name", esc(f.name)) + span("count", f.count) + (submenus ? svg("next", 12) : "");
		} else {
			if (f === current) cls += " selected";
			// file names line up with the folder names/icons next to them
			pad = row.path ? 0.5 : submenus ? 1.625 : 1.25 + row.depth;
			const fav = picks.fav.includes(String(f.value));
			// LTR isolate so the rtl clipping doesn't reorder the path segments
			html = span("name", mark(f.name)) + (row.path && f.parent.parent ? span("path", `\u2066${mark(f.parent.path)}\u2069`) : "")
				+ span("end", (byDate && f.mtime ? span("age", ago(f.mtime)) : "") + (f === current ? svg("check", 14, "kj-ctree-check") : "")
					+ `<span class="kj-ctree-star${fav ? " on" : ""}" title="${fav ? "Remove from" : "Add to"} favorites">${svg("star", 13)}</span>`);
		}
		return `<div class="${cls}" data-i="${i}" style="padding-left:${pad}rem"${row.file ? ` title="${esc(f.value)}"` : ""}>${html}</div>`;
	};

	// only the rows in view (plus a margin) are in the DOM, so cost doesn't grow with the list
	const paint = (p) => {
		const first = Math.max(0, Math.floor(p.list.scrollTop / rowH) - 10);
		p.start = first;
		p.win.style.transform = `translateY(${first * rowH}px)`;
		p.win.innerHTML = p.rows.slice(first, first + Math.ceil(p.list.clientHeight / rowH) + 20).map((r, k) => rowHtml(p, r, first + k)).join("");
	};

	// group headers aren't selectable: skip them in the direction of travel, or back from the list edge
	const setActive = (p, i, scroll = true, dir = 1) => {
		if (!p.rows.length) return;
		i = clamp(i, 0, p.rows.length - 1);
		let j = i;
		while ("head" in (p.rows[j] ?? {})) j += dir;
		if (!p.rows[j]) for (j = i; "head" in (p.rows[j] ?? {}); j -= dir);
		if (!p.rows[j]) return;
		p.win.querySelector(".active")?.classList.remove("active");
		p.active = j;
		p.win.children[j - p.start]?.classList.add("active");
		if (!scroll) return;
		// scroll the row into view by index (list padding is 0.25rem), then repaint the window there
		const top = 0.25 * rem + j * rowH, l = p.list;
		if (top < l.scrollTop) l.scrollTop = top;
		else if (top + rowH > l.scrollTop + l.clientHeight) l.scrollTop = top + rowH - l.clientHeight;
		paint(p);
	};

	// favorites, then recent picks not already favorited, above the folders of the first panel
	const groupRows = () => {
		const files = (values) => values.map((v) => byValue.get(v)).filter(Boolean);
		const fav = files(picks.fav);
		const recent = files(picks.recent).filter((f) => !fav.includes(f)).slice(0, 6);
		const group = (head, list) => (list.length ? [{ head }, ...list.map((file) => ({ file, depth: 0, path: true }))] : []);
		const rows = [...group("Favorites", fav), ...group("Recent", recent)];
		return rows.length ? [...rows, { head: "All" }] : rows;
	};

	const render = (p, item) => {
		const searching = p.level === 0 && query;
		const key = searching ? pathKey(query) : "";
		const terms = searching ? key.split(/\s+/) : null;
		p.hit = terms && new RegExp(`(${terms.map(reEsc).join("|")})`, "i");
		if (terms) {
			// typing more onto the previous query only narrows it, so filter the previous hits instead of every file
			p.hits = searchFiles(p.hitsKey && key.startsWith(p.hitsKey) ? p.hits : tree.all, terms, byDate);
			p.rows = p.hits.map((file) => ({ file, depth: 0, path: true }));
		} else p.rows = [...(p.level ? [] : groupRows()), ...treeRows(p.dir, expanded, p.level, byDate)];
		p.hitsKey = key;
		// drop the old rows first so the list clamps its scroll to the new height; setActive below repaints
		p.win.textContent = "";
		p.sizer.style.height = `${p.rows.length * rowH}px`;
		p.note.hidden = p.rows.length > 0;
		// a long submenu gets a fixed width, since only part of it is rendered to size it by
		if (p.level) p.el.style.width = p.rows.length > 40 ? "26rem" : "";
		// never shrink while open, so a bottom-anchored box doesn't slide down as the search filters it
		if (!p.level) p.el.style.minHeight = `${p.el.offsetHeight}px`;
		let i = p.rows.findIndex((r) => (r.folder ?? r.file) === item);
		// otherwise land on the current value or the folder holding it
		if (i === -1 && !searching) i = p.rows.findIndex((r) => r.file === current || holders.has(r.folder));
		setActive(p, Math.max(i, 0));
	};
	const rerender = (p) => render(p, p.rows[p.active]?.folder ?? p.rows[p.active]?.file);

	const toggleFav = (p, i) => {
		const v = String(p.rows[i].file.value);
		picks.fav = picks.fav.includes(v) ? picks.fav.filter((x) => x !== v) : [v, ...picks.fav];
		savePicks();
		rerender(p);
		// the favorites group lives in the first panel
		if (p !== panels[0]) {
			rerender(panels[0]);
			markOpen(panels[0]);
		}
	};

	const loadDates = async () => {
		if (datesLoaded) return;
		datesLoaded = true;
		try {
			const res = await api.fetchApi("/kjnodes/model_mtimes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values: tree.all.map((f) => f.value) }) });
			const times = await res.json();
			for (const f of tree.all) f.mtime = times[f.value] ?? 0;
			// a folder sorts by its newest file
			const newest = (d) => (d.mtime = Math.max(0, ...d.files.map((f) => f.mtime), ...[...d.folders.values()].map(newest)));
			newest(tree);
		} catch {
			return;
		}
		if (!ac.signal.aborted && byDate) {
			closeFrom(1);
			rerender(panels[0]);
		}
	};

	const toggle = (folder) => {
		expanded.delete(folder) || expanded.add(folder);
		render(panels[0], folder);
	};

	// repaint so the row whose submenu is open gets highlighted
	const markOpen = paint;

	const closeFrom = (level) => {
		if (panels.length <= level) return;
		while (panels.length > level) panels.pop().el.remove();
		focus = Math.min(focus, level - 1);
		markOpen(panels[level - 1]);
	};

	const openSub = (p, i, takeFocus) => {
		const folder = p.rows[i]?.folder;
		if (!folder) return;
		if (panels[p.level + 1]?.dir !== folder) {
			closeFrom(p.level + 1);
			const sub = makePanel(folder);
			Object.assign(sub.el.style, { visibility: "hidden", maxHeight: `${innerHeight - 16}px` });
			render(sub);
			// right of the parent panel, or left when it would leave the screen; aligned to the folder row
			const pr = p.el.getBoundingClientRect();
			const w = sub.el.offsetWidth;
			const top = Math.max(8, Math.min((p.win.children[i - p.start] ?? p.list).getBoundingClientRect().top - 5, innerHeight - 8 - sub.el.offsetHeight));
			Object.assign(sub.el.style, { left: `${pr.right - 4 + w <= innerWidth - 8 ? pr.right - 4 : Math.max(8, pr.left - w + 4)}px`, top: `${top}px`, visibility: "" });
			markOpen(p);
		}
		if (takeFocus) focus = p.level + 1;
	};

	const activate = (p, i) => {
		const row = p.rows[i];
		if (row?.file) {
			close();
			const v = String(row.file.value);
			picks.recent = [v, ...picks.recent.filter((x) => x !== v)].slice(0, 50);
			savePicks();
			applyValue(combo, row.file.value);
		} else if (submenus) openSub(p, i, true);
		else if (row) toggle(row.folder);
	};

	const hover = (p, i) => {
		focus = p.level;
		if (i !== p.active) setActive(p, i, false);
		if (!submenus || (p.level === 0 && query)) return;
		clearTimeout(hoverTimer);
		// short delay so a diagonal move toward an open submenu doesn't switch it on the way
		hoverTimer = setTimeout(() => (p.rows[i]?.folder ? openSub(p, i, false) : closeFrom(p.level + 1)), 200);
	};

	function makePanel(dir) {
		const el = h("div", { className: `kj-ctree${submenus ? " kj-ctree-menus" : ""}${panels.length ? " kj-ctree-sub" : ""}` });
		const list = el.appendChild(h("div", { className: "kj-ctree-list" }));
		list.style.maxHeight = `${maxRows * 1.75 + 0.5}rem`;
		// sizer holds the full list height, win the rendered rows shifted to the scroll position
		const sizer = list.appendChild(h("div"));
		const p = { el, list, sizer, win: sizer.appendChild(h("div")), dir, rows: [], active: 0, start: 0, level: panels.length, hit: null };
		p.note = list.appendChild(h("div", { className: "kj-ctree-note", textContent: "No results" }));
		list.addEventListener("scroll", () => paint(p), { passive: true });
		// keep focus in the search box when clicking rows or padding
		el.addEventListener("mousedown", (e) => e.target !== input && e.preventDefault());
		const onRow = (fn) => (e) => {
			const r = e.target.closest(".kj-ctree-row");
			if (r) fn(p, +r.dataset.i, e);
		};
		list.addEventListener("pointermove", onRow(hover));
		list.addEventListener("click", onRow((p, i, e) => (e.target.closest(".kj-ctree-star") ? toggleFav(p, i) : activate(p, i))));
		panels.push(p);
		document.body.appendChild(el);
		return p;
	}

	const onOutside = (e) => {
		if (panels.some((p) => p.el.contains(e.target))) return;
		// clicking the same combo's opener closes instead of reopening, like the native toggle
		if (e.type === "pointerdown" && e.target.closest?.(OPENERS)?.parentElement === anchor) reopenGuard = { anchor, t: performance.now() };
		close();
	};
	const close = () => {
		if (ac.signal.aborted) return;
		ac.abort();
		clearTimeout(hoverTimer);
		panels.forEach((p) => p.el.remove());
		closeTree = null;
	};

	const root = makePanel(tree);
	root.el.insertAdjacentHTML("afterbegin", `<label class="kj-ctree-search">${svg("search")}<input type="text" placeholder="Search" spellcheck="false">`
		+ `<button type="button" class="kj-ctree-sort${byDate ? " on" : ""}" title="Sort by date modified">${svg("clock")}</button></label>`);
	const input = root.el.querySelector("input");
	root.el.querySelector(".kj-ctree-sort").addEventListener("click", (e) => {
		byDate = !byDate;
		picks.sort = byDate ? "date" : "name";
		savePicks();
		e.currentTarget.classList.toggle("on", byDate);
		if (byDate) loadDates();
		closeFrom(1);
		rerender(root);
	});
	input.addEventListener("input", () => {
		query = input.value.trim();
		closeFrom(1);
		root.list.scrollTop = 0;
		render(root, query ? null : current);
	});
	input.addEventListener("keydown", (e) => {
		// Enter/arrows during IME composition belong to the IME
		if (e.isComposing) return;
		const p = panels[focus];
		const row = p.rows[p.active];
		if (MOVES[e.key]) setActive(p, p.active + MOVES[e.key], true, Math.sign(MOVES[e.key]));
		else if (e.key === "Enter") activate(p, p.active);
		else if (e.key === "Escape") close();
		else if (e.key === "ArrowRight" && !query && row?.folder) {
			if (!submenus && expanded.has(row.folder)) setActive(p, p.active + 1);
			else activate(p, p.active);
		} else if (e.key === "ArrowLeft" && !query && row) {
			const parent = (row.folder ?? row.file).parent;
			if (submenus) closeFrom(focus || 1);
			else if (row.folder && expanded.has(row.folder)) toggle(row.folder);
			else if (parent?.parent && !row.path) setActive(p, p.rows.findIndex((r) => r.folder === parent));
		} else return;
		e.preventDefault();
	});
	// keep graph keybindings out of the search box
	for (const t of ["keydown", "keyup"]) root.el.addEventListener(t, (e) => e.stopPropagation());

	// sized for the tallest the list can get (all folders open / search), capped by the setting
	const { r, side, space } = placement(anchor, (Math.min(maxRows, tree.all.length + tree.folderCount) * 1.75 + 3) * rootRem());
	const width = Math.min(Math.max(r.width, 480), innerWidth - 16);
	Object.assign(root.el.style, {
		left: `${clamp(r.left, 8, innerWidth - width - 8)}px`,
		width: `${width}px`,
		maxHeight: `${space}px`,
		...(side === "below" ? { top: `${r.bottom + 2}px` } : { bottom: `${side === "above" ? innerHeight - r.top + 2 : 8}px` }),
	});
	render(root, current);
	input.focus({ preventScroll: true });
	if (byDate) loadDates();

	// pointerdown/wheel elsewhere, focus moving outside (Tab) or resize dismiss it, like reka; window blur keeps it open
	const opts = { capture: true, signal: ac.signal };
	window.addEventListener("pointerdown", onOutside, opts);
	window.addEventListener("wheel", onOutside, { ...opts, passive: true });
	document.addEventListener("focusin", onOutside, opts);
	window.addEventListener("resize", close, opts);
	closeTree = close;
}

window.addEventListener("click", (e) => {
	const folders = folderMode !== "disabled";
	if (!maxVisible && !folders) return;
	const anchor = e.target.closest?.(OPENERS)?.parentElement;
	const trigger = anchor?.querySelector(`:scope > ${TRIGGER}`);
	if (!trigger) return;
	const combo = resolveCombo(trigger);
	// labelled options (getOptionLabel) aren't paths, keep them on the native list
	if (!folders || !combo || combo.widget.options?.getOptionLabel || !(combo.values.some((v) => PATH_FILE.test(String(v))) || (combo.values.length > LONG_LIST && nativeListIsSlow()))) {
		if (maxVisible) placeNative(anchor, combo?.values.length);
		return;
	}
	// swallow the click before reka's trigger handler opens the flat list
	e.preventDefault();
	e.stopPropagation();
	const guarded = reopenGuard?.anchor === anchor && performance.now() - reopenGuard.t < 1000;
	reopenGuard = null;
	if (!guarded) openFolderTree(anchor, combo);
}, true);

const setting = (key, name, tooltip, rest) => ({ id: `KJNodes.vueCombo.${key}`, name, category: ["KJNodes", "Nodes 2.0", name], tooltip, ...rest });

app.registerExtension({
	name: "KJNodes.VueCombo",
	settings: [
		setting("maxVisible", "Dropdown visible items", "How many options Nodes 2.0 combo dropdowns show before scrolling. 0 keeps the frontend default (7; 15 for the folder view).", {
			type: "slider",
			defaultValue: 0,
			attrs: { min: 0, max: 100, step: 1 },
			onChange: (value) => {
				maxVisible = value ?? 0;
				docEl.style.setProperty("--kj-combo-rows", maxVisible);
				docEl.toggleAttribute("data-kj-combo-rows", maxVisible > 0);
				if (!maxVisible) docEl.removeAttribute("data-kj-combo-pin");
			},
		}),
		setting("arrows", "Dropdown prev/next arrows", "Add legacy-style arrow buttons to Nodes 2.0 combo widgets for stepping to the previous/next option.", {
			type: "boolean",
			defaultValue: false,
			onChange: (value) => setArrows(!!value),
		}),
		setting("folders", "Dropdown folders", "Show Nodes 2.0 combos whose options contain subfolders (models, LoRAs...) as a searchable folder view: 'tree' expands folders inline, 'submenus' opens them to the side.", {
			type: "combo",
			options: ["disabled", "tree", "submenus"],
			defaultValue: "disabled",
			onChange: (value) => {
				folderMode = ["tree", "submenus"].includes(value) ? value : "disabled";
				closeTree?.();
			},
		}),
	],
});
