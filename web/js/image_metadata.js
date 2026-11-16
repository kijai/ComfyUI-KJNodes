const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const HEAD_BYTES = 64 * 1024;
const EBML_BYTES = 2 * 1024 * 1024; // the core webm/mkv parser only reads this much
const MAX_BOX = 64 * 1024 * 1024;
let closeActive = null;
let showSeq = 0;

function el(tag, props = {}, ...children) {
	const e = Object.assign(document.createElement(tag), props);
	e.append(...children.filter(Boolean));
	return e;
}

document.head.append(el("style", { textContent: `
.kj-meta-overlay { position: fixed; inset: 0; z-index: 10000; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
.kj-meta { background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd); border: 1px solid var(--border-color, #444); border-radius: 8px; width: min(860px, calc(100vw - 32px)); max-height: 85vh; display: flex; flex-direction: column; box-shadow: 0 8px 32px rgba(0,0,0,0.5); font-size: 13px; }
.kj-meta-header, .kj-meta-footer { display: flex; align-items: center; gap: 8px; padding: 10px 14px; }
.kj-meta-header { border-bottom: 1px solid var(--border-color, #444); }
.kj-meta-footer { border-top: 1px solid var(--border-color, #444); justify-content: flex-end; }
.kj-meta-title, .kj-meta-label { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kj-meta-title { font-weight: 600; }
.kj-meta-dim, .kj-meta-to, .kj-meta-empty, .kj-meta summary { color: var(--descrip-text, #999); font-weight: normal; }
.kj-meta-body { overflow-y: auto; padding: 10px 14px; display: flex; flex-direction: column; gap: 12px; }
.kj-meta-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.kj-meta textarea { width: 100%; box-sizing: border-box; resize: vertical; background: var(--comfy-input-bg, #111); color: var(--input-text, #ddd); border: 1px solid var(--border-color, #444); border-radius: 4px; padding: 6px; font-family: monospace; font-size: 12px; }
.kj-meta button { background: var(--comfy-input-bg, #333); color: var(--input-text, #ddd); border: 1px solid var(--border-color, #555); border-radius: 4px; padding: 3px 10px; cursor: pointer; font-size: 12px; }
.kj-meta button:hover { filter: brightness(1.25); }
.kj-meta summary { cursor: pointer; margin-bottom: 6px; }
` }));

// /view basenames filename, so a "sub/clip.mp4" value needs its folder moved into subfolder
function viewUrl(filename, type, subfolder = "") {
	const i = filename.lastIndexOf("/");
	if (i >= 0 && !subfolder) [subfolder, filename] = [filename.slice(0, i), filename.slice(i + 1)];
	return new URL(api.apiURL(`/view?${new URLSearchParams({ filename, type, subfolder })}`), location.href);
}

// header size only when the element shows the file itself, not a downscaled/HDR preview asset
function sizeOf(el) {
	if (!el || el.dataset?.fileUrl || /\/vhs\/viewvideo/.test(el.src ?? "")) return "";
	const w = el.videoWidth || el.naturalWidth;
	return w ? `${w}×${el.videoHeight || el.naturalHeight}` : "";
}

// VHS keeps its own preview widget; core images live in node.imgs, core videos in node.videoContainer
function mediaSource(node) {
	const vhs = node.widgets?.find((w) => w.name === "videopreview");
	const p = vhs?.value?.params;
	if (p?.filename && ["input", "output", "temp"].includes(p.type)) {
		return { url: viewUrl(p.filename, p.type, p.subfolder), size: sizeOf(vhs.videoEl?.hidden ? vhs.imgEl : vhs.videoEl), video: !p.format?.startsWith("image") };
	}
	const container = node.videoContainer?.querySelector("video[src]");
	const img = node.imgs?.[node.imageIndex ?? node.overIndex ?? 0] ?? node.imgs?.[0];
	const media = node.previewMediaType === "video" ? container : img ?? container;
	if (!media) return null;
	// preview/channel params make /view re-encode the file, which drops the metadata
	const url = new URL(media.dataset?.fileUrl ?? media.src, media.baseURI);
	url.searchParams.delete("preview");
	url.searchParams.delete("channel");
	return { url, size: sizeOf(media), video: media instanceof HTMLVideoElement };
}

// the core parser dispatches on MIME type, and /view may serve avif etc. as octet-stream
function sniffType(bytes, fallback) {
	const ascii = (a, b) => String.fromCharCode(...bytes.subarray(a, b));
	if (bytes[0] === 0x89 && ascii(1, 4) === "PNG") return "image/png";
	if (ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") return "image/webp";
	if (ascii(4, 8) === "ftyp") return /avi[fs]/.test(ascii(8, Math.min(boxAt(bytes, 0).size, bytes.length))) ? "image/avif" : "video/mp4";
	// old QuickTime files can start without ftyp
	if (/^(moov|mdat|wide|free|skip)$/.test(ascii(4, 8))) return "video/mp4";
	if (bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3) return "video/webm";
	return fallback;
}

// a server or proxy that ignores Range sends the whole file, so cap what gets buffered
async function readCapped(res, cap) {
	const reader = res.body.getReader();
	const chunks = [];
	for (let n = 0; n < cap; ) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		n += value.length;
	}
	reader.cancel().catch(() => {});
	return new Uint8Array(await new Blob(chunks).arrayBuffer()).subarray(0, cap);
}

async function fetchBytes(url, start, end) {
	const res = await fetch(url, start == null ? {} : { headers: { Range: `bytes=${start}-${end}` } });
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	const total = Number(res.headers.get("content-range")?.split("/")[1]);
	const buf = res.status === 206 ? new Uint8Array(await res.arrayBuffer()) : await readCapped(res, MAX_BOX);
	return { buf, status: res.status, total, type: res.headers.get("content-type") || "" };
}

// ISOBMFF box header at o; size 0 means "to end", size 1 a 64-bit size
function boxAt(b, o, end = b.length) {
	const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
	let size = dv.getUint32(o);
	let h = 8;
	if (size === 1 && o + 16 <= b.length) [size, h] = [Number(dv.getBigUint64(o + 8)), 16];
	else if (size === 0) size = end - o;
	return { kind: String.fromCharCode(...b.subarray(o + 4, o + 8)), id: dv.getUint32(o + 4), h, size };
}

function boxList(b, start, end) {
	const out = [];
	for (let o = start; o + 8 <= end; ) {
		const x = boxAt(b, o, end);
		if (x.size < x.h || o + x.size > end) break;
		out.push({ ...x, start: o + x.h, end: o + x.size });
		o += x.size;
	}
	return out;
}
const childBox = (b, box, kind) => boxList(b, box.start, box.end).find((x) => x.kind === kind);

// walks top-level boxes with range reads and keeps moov (or a top-level udta), where the comfy keys live
async function isobmffMetaBoxes(url, head) {
	const read = async (start, len) => start + len <= head.buf.length
		? head.buf.subarray(start, start + len)
		: (await fetchBytes(url, start, Math.min(start + len, head.total) - 1)).buf;
	const parts = [];
	for (let off = 0, n = 0; off + 8 <= head.total && n < 256; n++) {
		const { kind, size } = boxAt(await read(off, 16), 0, head.total - off);
		if (size < 8) break;
		if ((kind === "moov" || kind === "udta") && size <= MAX_BOX) {
			parts.push(await read(off, size));
			if (kind === "moov") break;
		}
		off += size;
	}
	return parts;
}

// videos can be huge, so only fetch the parts the parsers look at
async function loadParts(url) {
	let head = await fetchBytes(url, 0, HEAD_BYTES - 1);
	// 206 without a readable Content-Range (e.g. not CORS-exposed): one capped plain read instead
	if (head.status === 206 && !Number.isFinite(head.total)) head = await fetchBytes(url);
	const type = sniffType(head.buf, head.type);
	if (head.status !== 206 || head.buf.length >= head.total) return { parts: [head.buf], type };
	if (type === "video/mp4") return { parts: await isobmffMetaBoxes(url, head), type };
	const end = type.startsWith("image/") ? head.total - 1 : Math.min(EBML_BYTES, head.total) - 1;
	return { parts: [head.buf, (await fetchBytes(url, head.buf.length, end)).buf], type };
}

const utf8 = (b, start, end) => new TextDecoder().decode(b.subarray(start, end));

// raw udta/meta mdta strings, keyed by lowercase name
function isobmffTags(b) {
	const moov = childBox(b, { start: 0, end: b.length }, "moov");
	const udta = childBox(b, { start: 0, end: b.length }, "udta") ?? (moov && childBox(b, moov, "udta"));
	const meta = udta && childBox(b, udta, "meta");
	const body = meta && { start: meta.start + 4, end: meta.end };
	const keys = body && childBox(b, body, "keys");
	const ilst = body && childBox(b, body, "ilst");
	if (!keys || !ilst) return {};
	// keys entries are box-shaped: size, namespace, name
	const names = boxList(b, keys.start + 8, keys.end).map((k) => utf8(b, k.start, k.end));
	const tags = {};
	for (const item of boxList(b, ilst.start, ilst.end)) {
		const data = childBox(b, item, "data");
		const name = names[item.id - 1];
		if (data && name) tags[name.toLowerCase()] = utf8(b, data.start + 8, data.end);
	}
	return tags;
}

// Matroska SimpleTag: TagName (45 A3) then TagString (44 87) within a few elements; names are checked, so a byte scan is enough
function ebmlTags(b) {
	const vint = (p) => {
		let len = 1;
		while (len <= 8 && !(b[p] & (0x80 >> (len - 1)))) len++;
		if (len > 8 || p + len > b.length) return null;
		let v = b[p] & (0xff >> len);
		for (let i = 1; i < len; i++) v = v * 256 + b[p + i];
		return { len, v };
	};
	const tags = {};
	for (let p = 0; p + 4 < b.length; p++) {
		if (b[p] !== 0x45 || b[p + 1] !== 0xa3) continue;
		const ns = vint(p + 2);
		if (!ns || ns.v > 32) continue;
		const nameEnd = p + 2 + ns.len + ns.v;
		const name = utf8(b, p + 2 + ns.len, nameEnd).toLowerCase();
		if (name !== "prompt" && name !== "workflow") continue;
		for (let q = nameEnd; q < nameEnd + 32 && q + 3 < b.length; q++) {
			if (b[q] !== 0x44 || b[q + 1] !== 0x87) continue;
			const vs = vint(q + 2);
			if (vs) tags[name] = utf8(b, q + 2 + vs.len, Math.min(q + 2 + vs.len + vs.v, b.length));
			break;
		}
	}
	return tags;
}

async function readMetadata(url) {
	const { parts, type } = await loadParts(url);
	const name = url.searchParams.get("filename") || "file";
	// own raw tag readers for video: the core ones only accept values starting with "{", which drops VHS's double-encoded prompt
	if (type.startsWith("video/")) {
		const b = parts.length === 1 ? parts[0] : new Uint8Array(await new Blob(parts).arrayBuffer());
		return { name, raw: type === "video/mp4" ? isobmffTags(b) : ebmlTags(b), video: true };
	}
	const parse = window.comfyAPI?.parser?.getWorkflowDataFromFile;
	if (!parse) throw new Error("this ComfyUI frontend doesn't expose comfyAPI.parser");
	return { name, raw: (await parse(new File(parts, name, { type }))) || {}, video: false };
}

// VHS double-encodes prompt, and python's json.dumps writes bare NaN/Infinity (e.g. "is_changed": NaN), which JSON.parse rejects
function asObject(v) {
	for (let i = 0; i < 3 && typeof v === "string"; i++) {
		try {
			v = JSON.parse(v.replace(/"(?:[^"\\]|\\.)*"|-?\b(?:NaN|Infinity)\b/g, (m) => (m[0] === '"' ? m : "null")));
		} catch {
			return null;
		}
	}
	return v && typeof v === "object" ? v : null;
}

const isLink = (v) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && typeof v[1] === "number";

// live node defs tell prompt text apart from combo values; unknown (uninstalled) classes fall back to a heuristic
function isPromptText(classType, name, value) {
	const def = LiteGraph.registered_node_types?.[classType]?.nodeData?.input;
	const spec = def?.required?.[name] ?? def?.optional?.[name];
	// "wordy" keeps number lists like ManualSigmas' "1.0, 0.99, ..." out
	const wordy = /\s/.test(value) && /\p{L}{2}/u.test(value);
	if (spec) return spec[0] === "STRING" && (!!spec[1]?.multiline || wordy);
	return value.length >= 20 && wordy;
}

function extract(prompt) {
	const titleOf = (id) => prompt[id]?._meta?.title || prompt[id]?.class_type || id;
	const consumers = {};
	for (const [id, n] of Object.entries(prompt)) {
		for (const [k, v] of Object.entries(n?.inputs ?? {})) {
			if (isLink(v)) (consumers[v[0]] ??= new Set()).add(`${titleOf(id)}.${k}`);
		}
	}
	const texts = [];
	const others = [];
	for (const [id, n] of Object.entries(prompt)) {
		const title = titleOf(id);
		const rest = [];
		for (const [k, v] of Object.entries(n?.inputs ?? {})) {
			if (v == null || isLink(v)) continue;
			if (typeof v === "string" && v.trim() && isPromptText(n.class_type, k, v)) texts.push({ label: `${title} #${id} · ${k}`, to: [...(consumers[id] ?? [])], text: v });
			else rest.push(`  ${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
		}
		if (rest.length) others.push(`#${id} ${title}${n.class_type && title !== n.class_type ? ` (${n.class_type})` : ""}\n${rest.join("\n")}`);
	}
	texts.sort((a, b) => b.text.length - a.text.length);
	return { texts, others: others.join("\n\n") };
}

function copyButton(label, getText) {
	let timer;
	const b = el("button", { textContent: label });
	b.onclick = async () => {
		const text = getText();
		let ok = true;
		try {
			await navigator.clipboard.writeText(text);
		} catch {
			// navigator.clipboard is missing on non-secure origins (LAN http)
			const ta = document.body.appendChild(el("textarea", { value: text }));
			ta.style.cssText = "position:fixed;opacity:0;";
			ta.select();
			ok = document.execCommand("copy");
			ta.remove();
		}
		b.textContent = ok ? "Copied" : "Copy failed";
		clearTimeout(timer);
		timer = setTimeout(() => (b.textContent = label), 1000);
	};
	return b;
}

const textBox = (value, rows) => el("textarea", { value, rows, readOnly: true, spellcheck: false });

function showDialog({ name, size, texts, others, prompt, workflow }) {
	closeActive?.();
	const body = el("div", { className: "kj-meta-body" },
		...texts.map((t) => el("div", {},
			el("div", { className: "kj-meta-head" },
				el("span", { className: "kj-meta-label", title: t.label, textContent: t.label },
					t.to?.length && el("span", { className: "kj-meta-to", textContent: `  → ${t.to.join(", ")}` })),
				copyButton("Copy", () => t.text)),
			textBox(t.text, 2))),
		others && el("details", {}, el("summary", { textContent: "Other inputs" }), textBox(others, Math.min(20, others.split("\n").length))),
		!texts.length && !others && el("div", { className: "kj-meta-empty", textContent: "No ComfyUI prompt metadata found." }));

	const close = () => {
		overlay.remove();
		window.removeEventListener("keydown", onKey, true);
		if (closeActive === close) closeActive = null;
	};
	// window capture runs before the canvas/keybinding handlers, so Delete etc. can't act on the graph behind
	const onKey = (e) => {
		e.stopPropagation();
		if (e.key === "Escape") {
			e.preventDefault();
			close();
		}
	};
	const overlay = el("div", { className: "kj-meta-overlay", onpointerdown: (e) => e.target === overlay && close() },
		el("div", { className: "kj-meta" },
			el("div", { className: "kj-meta-header" },
				el("span", { className: "kj-meta-title", title: name, textContent: name }, size && el("span", { className: "kj-meta-dim", textContent: `  ${size}` })),
				el("button", { textContent: "Close", onclick: close })),
			body,
			el("div", { className: "kj-meta-footer" },
				prompt && copyButton("Copy API prompt", () => JSON.stringify(prompt, null, 2)),
				workflow && copyButton("Copy workflow", () => JSON.stringify(workflow)))));

	document.body.append(overlay);
	// fit to content (wrapping included) once laid out; long prompts scroll inside the box
	for (const ta of body.querySelectorAll(":scope > div textarea")) ta.style.height = `${Math.min(ta.scrollHeight + 2, 260)}px`;
	window.addEventListener("keydown", onKey, true);
	closeActive = close;
}

async function showMetadata(src) {
	const seq = ++showSeq;
	try {
		const { name, raw, video } = await readMetadata(src.url);
		if (seq !== showSeq) return; // a later request owns the dialog
		const prompt = asObject(raw.prompt ?? raw.Prompt);
		const workflow = asObject(raw.workflow ?? raw.Workflow);
		const { texts, others } = prompt ? extract(prompt) : { texts: [], others: "" };
		// raw text: image chunks like A1111 "parameters", plus prompt/workflow that didn't parse; video container tags are noise
		for (const [k, v] of Object.entries(raw)) {
			if (v == null || (/^(prompt|workflow)$/i.test(k) ? asObject(v) : video)) continue;
			const text = typeof v === "string" ? v : JSON.stringify(v, null, 2);
			if (text.trim()) texts.push({ label: k, text });
		}
		showDialog({ name, size: src.size, texts, others, prompt, workflow });
	} catch (err) {
		if (seq !== showSeq) return;
		console.error("[KJNodes.ImageMetadata]", err);
		app.extensionManager?.toast?.add({ severity: "error", summary: "Failed to read metadata", detail: String(err?.message ?? err), life: 4000 });
	}
}

app.registerExtension({
	name: "KJNodes.ImageMetadata",
	getNodeMenuItems(node) {
		// save/preview nodes only hold the workflow that's already open; this is for files being loaded
		if (node.constructor?.nodeData?.output_node) return [];
		const src = mediaSource(node);
		return src ? [{ content: src.video ? "View Video Metadata" : "View Image Metadata", callback: () => showMetadata(src) }] : [];
	},
});
