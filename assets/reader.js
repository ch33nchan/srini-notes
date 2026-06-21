const project = document.body.dataset.project || "openpilot";
const manifestPath = `./manifest.json`;
const noteList = document.querySelector("#note-list");
const content = document.querySelector("#note-content");
const toc = document.querySelector("#toc");
const sidebar = document.querySelector("#reader-sidebar");
const toggle = document.querySelector(".sidebar-toggle");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

function rewriteHref(href) {
  return href;
}

function inlineMarkdown(text) {
  let value = escapeHtml(text);
  value = value.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_, alt, href) => {
    return `<img alt="${escapeHtml(alt)}" src="${escapeHtml(href)}">`;
  });
  value = value.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, href) => {
    const rewritten = rewriteHref(href);
    return `<a href="${escapeHtml(rewritten)}" target="${rewritten.startsWith("http") ? "_blank" : "_self"}" rel="noreferrer">${label}</a>`;
  });
  value = value.replace(/`([^`]+)`/g, "<code>$1</code>");
  value = value.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  return value;
}

function closeList(state, html) {
  if (state.list) {
    html.push(`</${state.list}>`);
    state.list = null;
  }
}

function renderMarkdown(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  const headings = [];
  const state = { code: false, codeLang: "", codeLines: [], list: null };

  function flushCode() {
    html.push(`<pre><code>${escapeHtml(state.codeLines.join("\n"))}</code></pre>`);
    state.code = false;
    state.codeLang = "";
    state.codeLines = [];
  }

  for (const line of lines) {
    const fence = line.match(/^```(.*)$/);
    if (fence) {
      if (state.code) {
        flushCode();
      } else {
        closeList(state, html);
        state.code = true;
        state.codeLang = fence[1].trim();
        state.codeLines = [];
      }
      continue;
    }

    if (state.code) {
      state.codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      closeList(state, html);
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList(state, html);
      const level = heading[1].length;
      const text = heading[2].trim();
      const id = slugify(text);
      headings.push({ level, text: text.replace(/`/g, ""), id });
      html.push(`<h${level} id="${id}">${inlineMarkdown(text)}</h${level}>`);
      continue;
    }

    const unordered = line.match(/^\s*[-*]\s+(.+)$/);
    if (unordered) {
      if (state.list !== "ul") {
        closeList(state, html);
        state.list = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${inlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (ordered) {
      if (state.list !== "ol") {
        closeList(state, html);
        state.list = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${inlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    const quote = line.match(/^>\s+(.+)$/);
    if (quote) {
      closeList(state, html);
      html.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    closeList(state, html);
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }

  if (state.code) flushCode();
  closeList(state, html);
  return { html: html.join("\n"), headings };
}

function renderToc(headings) {
  const usable = headings.filter((heading) => heading.level === 2 || heading.level === 3);
  toc.innerHTML = usable
    .map((heading) => `<a class="toc-h${heading.level}" href="#${heading.id}">${escapeHtml(heading.text)}</a>`)
    .join("");
}

async function loadNote(note, notes) {
  const response = await fetch(note.file);
  if (!response.ok) throw new Error(`Could not load ${note.file}`);
  const markdown = await response.text();
  const rendered = renderMarkdown(markdown);
  content.innerHTML = rendered.html;
  renderToc(rendered.headings);
  document.title = `${note.title} | OpenPilot Notes`;
  window.history.replaceState(null, "", `#${note.slug}`);
  document.querySelectorAll(".note-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.slug === note.slug);
  });
  if (sidebar) sidebar.classList.remove("open");
  if (toggle) toggle.setAttribute("aria-expanded", "false");
}

function renderNoteList(notes) {
  noteList.innerHTML = notes
    .map((note) => `
      <button class="note-button" type="button" data-slug="${note.slug}">
        <strong>${escapeHtml(note.title)}</strong>
        <span>${escapeHtml(note.summary)}</span>
      </button>
    `)
    .join("");

  noteList.querySelectorAll(".note-button").forEach((button) => {
    button.addEventListener("click", () => {
      const note = notes.find((candidate) => candidate.slug === button.dataset.slug);
      if (note) loadNote(note, notes);
    });
  });
}

async function init() {
  const response = await fetch(manifestPath);
  if (!response.ok) throw new Error("Could not load note manifest");
  const notes = await response.json();
  renderNoteList(notes);
  const slug = window.location.hash.replace("#", "");
  const selected = notes.find((note) => note.slug === slug) || notes[0];
  await loadNote(selected, notes);
}

if (toggle) {
  toggle.addEventListener("click", () => {
    const open = !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
  });
}

init().catch((error) => {
  content.innerHTML = `<p class="loading">${escapeHtml(error.message)}</p>`;
});
