/* AIO Local — vanilla JS UI. No build step, no third-party code. */
(() => {
  "use strict";

  const TOKEN = (document.querySelector('meta[name="aio-token"]') || {}).content || "";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const API = "/api/v1";

  const state = {
    settings: null,
    stats: {},
    models: null,
    conversations: [],
    conversationId: null,
    streaming: false,
    controller: null,
    chatImage: null,
    chatDocJob: null,
    chatDocTimer: null,
    selectedImage: null,
    gallery: [],
    view: "chat",
    paths: {},
    requirements: null,
    setupTimer: null,
    logTimer: null,
    codeEmbedUrl: "",
    codeEmbedBusy: false,
  };

  const PLANNED = {
    media: {
      title: "Audio & video",
      body: "Audio and video are deliberately deferred to v2. Video would use OpenVINO GenAI text-to-video pipelines, not an OVMS endpoint, so it needs its own worker and disk policy.",
      items: [
        "Speech-to-text (Whisper) and text-to-speech (Kokoro)",
        "LTX-Video text-to-video with strict disk limits",
        "Media artefacts listed in the gallery",
        "No cloud calls at any point",
      ],
    },
  };

  let modelsTimer = null;

  /* ------------------------------------------------------------ utilities */
  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeAttr(text) {
    return escapeHtml(text).replace(/`/g, "&#96;");
  }

  function fmtDate(ms) {
    if (!ms) return "";
    try {
      return new Date(Number(ms)).toLocaleString(undefined, {
        year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
      });
    } catch { return ""; }
  }

  function toast(message, isError = false) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.toggle("err", !!isError);
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), 3600);
  }

  async function api(path, options = {}) {
    const headers = Object.assign(
      { "X-AIO-Token": TOKEN },
      options.body ? { "Content-Type": "application/json" } : {},
      options.headers || {}
    );
    const url = path.startsWith("/api/") || path.startsWith("/v1/") ? path : API + path;
    const res = await fetch(url, {
      method: options.method || "GET",
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: options.signal,
    });
    let data = null;
    try { data = await res.json(); } catch { data = null; }
    if (!res.ok) {
      const message = (data && data.error && data.error.message) || `Request failed (HTTP ${res.status})`;
      const err = new Error(message);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  /* ------------------------------------------------------------ markdown */
  function inlineFmt(text) {
    const parts = String(text).split("`");
    let out = "";
    parts.forEach((part, index) => {
      if (index % 2 === 1) {
        out += `<code>${part}</code>`;
        return;
      }
      out += part
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
          '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    });
    return out;
  }

  function renderInline(text) {
    const lines = String(text).split("\n");
    let html = "";
    let list = null;
    let para = [];
    const flushPara = () => {
      if (para.length) { html += `<p>${para.join("<br>")}</p>`; para = []; }
    };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };

    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (!line.trim()) { flushPara(); closeList(); continue; }

      const h = line.match(/^(#{1,3})\s+(.*)$/);
      if (h) {
        flushPara(); closeList();
        const level = h[1].length + 2;
        html += `<h${level}>${inlineFmt(h[2])}</h${level}>`;
        continue;
      }
      const quote = line.match(/^&gt;\s?(.*)$/);
      if (quote) {
        flushPara(); closeList();
        html += `<blockquote>${inlineFmt(quote[1])}</blockquote>`;
        continue;
      }
      const ul = line.match(/^\s*[-*]\s+(.*)$/);
      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        const type = ul ? "ul" : "ol";
        if (list !== type) { closeList(); html += `<${type}>`; list = type; }
        html += `<li>${inlineFmt((ul || ol)[1])}</li>`;
        continue;
      }
      closeList();
      para.push(inlineFmt(line));
    }
    flushPara(); closeList();
    return html;
  }

  function renderMarkdown(text) {
    const parts = String(text).split("```");
    let out = "";
    parts.forEach((part, index) => {
      if (index % 2 === 1) {
        const nl = part.indexOf("\n");
        let lang = "";
        let code = part;
        if (nl >= 0) { lang = part.slice(0, nl).trim(); code = part.slice(nl + 1); }
        out += `<pre><code data-lang="${escapeAttr(lang)}">${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`;
      } else {
        const guarded = protectMath(part);
        out += restoreMath(renderInline(escapeHtml(guarded.out)), guarded.blocks);
      }
    });
    return out;
  }

  function protectMath(text) {
    const blocks = [];
    let out = String(text);
    const stash = (body, display) => {
      blocks.push({ body, display });
      return `\u0000M${blocks.length - 1}\u0000`;
    };
    out = out.replace(/\$\$([\s\S]+?)\$\$/g, (_, body) => stash(body, true));
    out = out.replace(/\\\[([\s\S]+?)\\\]/g, (_, body) => stash(body, true));
    out = out.replace(/\\\(([\s\S]+?)\\\)/g, (_, body) => stash(body, false));
    out = out.replace(/\$([^\n$]+?)\$/g, (_, body) => stash(body, false));
    return { out, blocks };
  }

  function restoreMath(html, blocks) {
    let out = html.replace(/<p>\s*\u0000M(\d+)\u0000\s*<\/p>/g, (match, index) => {
      const block = blocks[Number(index)];
      if (!block) return match;
      if (!block.display) return match;
      return `<div class="math-display">$$${escapeHtml(block.body)}$$</div>`;
    });
    return out.replace(/\u0000M(\d+)\u0000/g, (match, index) => {
      const block = blocks[Number(index)];
      if (!block) return match;
      return block.display
        ? `$$${escapeHtml(block.body)}$$`
        : `$${escapeHtml(block.body)}$`;
    });
  }

  function typesetMath(root) {
    if (!root || typeof window.renderMathInElement !== "function") return;
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
        ],
        throwOnError: false,
        strict: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      });
    } catch (_) { /* math rendering is optional */ }
  }

  /* ------------------------------------------------------------ navigation */
  const VIEW_META = {
    chat: ["Chat", "Memory-first conversation with your local model."],
    images: ["Images", "Generate and edit images locally through OVMS."],
    memory: ["Memory", "Inspect exactly what the assistant remembers and why."],
    models: ["Models", "Download models, watch progress and choose a device for each one."],
    documents: ["Knowledge", "Your private reference library — upload, scan, search. No fine-tuning."],
    code: ["Code", "Manage the local OpenCode server and browse its workspace."],
    re: ["Reverse engineering", "REA tools over MCP; every call is explicit."],
    gimp: ["GIMP dock", "Launch GIMP and hand it a generated image."],
    setup: ["Setup", "Portable components, install progress and complete logs."],
    settings: ["Settings", "Model server, context budget, module paths and local data."],
  };

  function showView(name) {
    state.view = name;
    $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === name));

    if (PLANNED[name]) {
      const info = PLANNED[name];
      $("#view-title").textContent = info.title;
      $("#view-sub").textContent = "Planned module — not installed in this build.";
      $$(".view").forEach((v) => v.classList.remove("active"));
      $("#view-planned").classList.add("active");
      $("#planned-title").textContent = info.title;
      $("#planned-body").textContent = info.body;
      $("#planned-list").innerHTML = info.items.map((i) => `<li>${escapeHtml(i)}</li>`).join("");
      return;
    }

    const meta = VIEW_META[name] || VIEW_META.chat;
    $("#view-title").textContent = meta[0];
    $("#view-sub").textContent = meta[1];
    $$(".view").forEach((v) => v.classList.remove("active"));
    const el = $(`#view-${name}`);
    if (el) el.classList.add("active");

    if (name === "memory") refreshMemory();
    if (name === "models") refreshModels();
    if (name === "images") refreshGallery();
    if (name === "settings") fillSettings();
    if (name === "documents") refreshKnowledge();
    if (name === "code") refreshCode();
    if (name === "re") refreshReverse();
    if (name === "gimp") refreshGimp();
    if (name === "setup") {
      refreshSetup();
      startLogPolling();
    } else {
      stopLogPolling();
    }
  }

  /* ------------------------------------------------------------ state/pills */
  function applyPills(payload) {
    const upstream = (payload.models && payload.models.upstream) || {};
    const pill = $("#upstream-pill");
    pill.classList.remove("offline", "online", "warn");
    if (upstream.ok) {
      pill.classList.add("online");
      pill.innerHTML = '<span class="dot"></span> model server online';
    } else {
      pill.classList.add("offline");
      pill.innerHTML = '<span class="dot"></span> OVMS offline';
    }

    const settings = payload.settings || state.settings || {};
    $("#model-pill").textContent = settings.chat_model || "no model";
    $("#mem-pill").textContent = settings.memory_enabled === false ? "memory off" : "memory on";

    if (payload.paths) state.paths = payload.paths;
  }

  async function refreshState() {
    try {
      const payload = await api("/state");
      state.settings = payload.settings;
      state.stats = payload.stats;
      state.models = payload.models;
      state.paths = payload.paths;
      applyPills(payload);
      populateChatModels();
      populateImageModels();
      if (state.view === "memory") renderMemoryStats();
      if (state.view === "models") renderModels();
      if (state.view === "settings") fillSettings();
    } catch (err) {
      if (err.status === 401) toast("Local token rejected. Restart the app.", true);
    }
  }

  /* ------------------------------------------------------------ chat */
  async function refreshConversations() {
    try {
      const payload = await api("/conversations");
      state.conversations = payload.data || [];
      const select = $("#conversation-select");
      const current = state.conversationId;
      select.innerHTML = '<option value="">New conversation…</option>' +
        state.conversations.map((c) => {
          const title = c.title || "Untitled chat";
          return `<option value="${escapeAttr(c.id)}">${escapeHtml(title)}</option>`;
        }).join("");
      if (current) select.value = current;
    } catch { /* non-fatal */ }
  }

  function clearMessages() {
    const box = $("#messages");
    box.innerHTML = "";
  }

  function showEmptyChat() {
    const box = $("#messages");
    box.innerHTML = `
      <div class="empty-state" id="chat-empty">
        <div class="empty-mark">◈</div>
        <h2>Start with anything</h2>
        <p>This assistant remembers across chats, on this machine only. Ask a question, paste code, or plan a project.</p>
        <div class="suggestions">
          <button class="chip" data-suggest="Summarise what you know about this project so far.">What do you remember?</button>
          <button class="chip" data-suggest="Help me plan a small Python tool, step by step.">Plan a small project</button>
          <button class="chip" data-suggest="Explain how the memory system in this app works.">How does memory work?</button>
        </div>
      </div>`;
  }

  function messageElement(role, content) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    const isUser = role === "user";
    wrap.innerHTML = `
      <div class="msg-avatar">${isUser ? "YOU" : "AI"}</div>
      <div class="msg-body">
        <div class="msg-role">${isUser ? "You" : "AIO Local"}</div>
        <div class="msg-content"></div>
        <div class="msg-meta"></div>
      </div>`;
    $(".msg-content", wrap).innerHTML = isUser ? `<p>${escapeHtml(content).replace(/\n/g, "<br>")}</p>` : renderMarkdown(content);
    if (!isUser) typesetMath($(".msg-content", wrap));
    return wrap;
  }

  function scrollMessages() {
    const box = $("#messages");
    box.scrollTop = box.scrollHeight;
  }

  function appendMessage(role, content) {
    const empty = $("#chat-empty");
    if (empty) empty.remove();
    const el = messageElement(role, content);
    $("#messages").appendChild(el);
    scrollMessages();
    return el;
  }

  function setAssistantContent(el, text, withMath = true) {
    const content = $(".msg-content", el);
    content.innerHTML = renderMarkdown(text);
    if (withMath) typesetMath(content);
    scrollMessages();
  }

  async function loadConversation(id) {
    state.conversationId = id || null;
    clearMessages();
    if (!id) { showEmptyChat(); return; }
    try {
      const payload = await api(`/conversations/${encodeURIComponent(id)}/messages`);
      const rows = payload.data || [];
      if (!rows.length) { showEmptyChat(); return; }
      for (const row of rows) {
        const el = appendMessage(row.role === "user" ? "user" : "assistant", row.content || "");
        const meta = $(".msg-meta", el);
        if (row.role !== "user") {
          meta.innerHTML = `<span class="muted small">${escapeHtml(fmtDate(row.created_at))}</span>`;
        }
      }
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function newConversation() {
    try {
      const payload = await api("/conversations", { method: "POST" });
      state.conversationId = payload.id;
      await refreshConversations();
      clearMessages();
      clearChatImage();
      showEmptyChat();
      $("#prompt").focus();
    } catch (err) {
      toast(err.message, true);
    }
  }

  function currentModelSupportsVision() {
    const models = (state.models && state.models.models) || [];
    const select = $("#chat-model");
    const id = (select && select.value) || (state.settings && state.settings.chat_model) || "";
    const entry = models.find((model) => model.id === id);
    return !!(entry && entry.vision);
  }

  function populateChatModels() {
    const select = $("#chat-model");
    if (!select) return;
    const models = ((state.models && state.models.models) || []).filter(
      (model) => model.task === "text_generation"
    );
    const current = select.value || (state.settings && state.settings.chat_model) || "";
    if (!models.length) {
      select.innerHTML = `<option value="${escapeAttr(current)}">${escapeHtml(current || "no model")}</option>`;
      return;
    }
    select.innerHTML = models.map((model) => {
      const tags = [model.vision ? "vision" : "", model.loaded ? "loaded" : ""].filter(Boolean).join(" · ");
      return `<option value="${escapeAttr(model.id)}" ${model.id === current ? "selected" : ""}>` +
        `${escapeHtml(model.name || model.id)}${tags ? " · " + escapeHtml(tags) : ""}</option>`;
    }).join("");
    if (!select.value && current) select.value = current;
  }

  function populateImageModels() {
    const models = ((state.models && state.models.models) || []).filter(
      (model) => model.task === "image_generation"
    );
    const ids = models.map((model) => model.id);
    const choose = (select, fallback) => {
      if (!select) return;
      const current = select.value || fallback;
      const options = ids.length ? ids : [fallback];
      select.innerHTML = options.map((id) =>
        `<option value="${escapeAttr(id)}" ${id === current ? "selected" : ""}>${escapeHtml(id)}</option>`
      ).join("");
      if (!select.value) select.value = fallback;
    };
    choose($("#img-model"), "sd15");
    choose($("#img-edit-model"), "sd15-inpaint");
  }

  function clearChatImage() {
    state.chatImage = null;
    const box = $("#chat-attachment");
    if (box) box.classList.add("hidden");
    const preview = $("#chat-attachment-preview");
    if (preview) preview.removeAttribute("src");
    const input = $("#chat-image-input");
    if (input) input.value = "";
  }

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("Could not read that image."));
      reader.readAsDataURL(file);
    });
  }

  function loadDataImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("That file is not a readable image."));
      img.src = src;
    });
  }

  async function prepareChatImage(file) {
    if (!file || !String(file.type || "").startsWith("image/")) {
      throw new Error("Choose an image file (PNG, JPEG, WebP or GIF).");
    }
    if (file.size > 25 * 1024 * 1024) throw new Error("Image is too large (25 MB max).");
    const source = await readFileAsDataURL(file);
    const img = await loadDataImage(source);
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    let edge = 1280;
    let quality = 0.85;
    let dataUrl = "";
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const scale = Math.min(1, edge / Math.max(width, height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      dataUrl = canvas.toDataURL("image/jpeg", quality);
      if (dataUrl.length <= 2_600_000) break;
      edge = Math.max(640, Math.round(edge * 0.7));
      quality = Math.max(0.6, quality - 0.1);
    }
    return { dataUrl, name: file.name || "image.jpg" };
  }

  async function attachChatImage(file) {
    try {
      const prepared = await prepareChatImage(file);
      state.chatImage = prepared;
      $("#chat-attachment-preview").src = prepared.dataUrl;
      $("#chat-attachment-name").textContent = prepared.name;
      $("#chat-attachment").classList.remove("hidden");
      $("#prompt").focus();
    } catch (err) {
      toast(err.message, true);
    } finally {
      $("#chat-image-input").value = "";
    }
  }

  async function sendMessage() {
    const promptEl = $("#prompt");
    const image = state.chatImage;
    const text = promptEl.value.trim() || (image ? "Describe this image." : "");
    if (!text || state.streaming) return;
    if (image && !currentModelSupportsVision()) {
      toast("Choose a vision model such as Qwen3-VL 4B in Models, then attach the image.", true);
      return;
    }
    if (!state.conversationId) {
      try {
        const payload = await api("/conversations", { method: "POST" });
        state.conversationId = payload.id;
        await refreshConversations();
      } catch (err) {
        toast(err.message, true);
        return;
      }
    }

    const userEl = appendMessage("user", text);
    if (image) {
      const thumb = document.createElement("img");
      thumb.className = "msg-image";
      thumb.src = image.dataUrl;
      thumb.alt = image.name || "attached image";
      $(".msg-content", userEl).prepend(thumb);
    }
    clearChatImage();
    promptEl.value = "";
    autoGrow(promptEl);

    const assistant = appendMessage("assistant", "");
    const contentEl = $(".msg-content", assistant);
    contentEl.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    const metaEl = $(".msg-meta", assistant);

    state.streaming = true;
    state.controller = new AbortController();
    $("#send").disabled = true;
    $("#stop").classList.remove("hidden");
    $("#chat-status").textContent = "thinking…";

    const body = {
      model: ($("#chat-model") && $("#chat-model").value) || (state.settings && state.settings.chat_model) || undefined,
      messages: [{
        role: "user",
        content: image
          ? [
              { type: "text", text },
              { type: "image_url", image_url: { url: image.dataUrl } },
            ]
          : text,
      }],
      stream: true,
    };

    let answer = "";
    let requestId = "";
    let memoryCount = 0;
    let finished = false;
    const headers = {
      "Content-Type": "application/json",
      "X-AIO-Token": TOKEN,
      "X-AIO-Conversation": state.conversationId,
    };
    const knowledge = $("#chat-knowledge") ? $("#chat-knowledge").value : "";
    if (knowledge) headers["X-AIO-Knowledge"] = knowledge;

    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: state.controller.signal,
      });

      requestId = res.headers.get("X-AIO-Request-Id") || "";
      memoryCount = parseInt(res.headers.get("X-AIO-Memory-Items") || "0", 10) || 0;

      if (!res.ok) {
        let message = `Request failed (HTTP ${res.status})`;
        try {
          const err = await res.json();
          if (err && err.error && err.error.message) message = err.error.message;
        } catch { /* ignore */ }
        throw new Error(message);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let index;
        while ((index = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, index).trim();
          buffer = buffer.slice(index + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          try {
            const obj = JSON.parse(data);
            const choice = (obj.choices && obj.choices[0]) || {};
            const delta = choice.delta || {};
            if (delta.content) {
              answer += delta.content;
              setAssistantContent(assistant, answer, false);
            }
          } catch { /* ignore malformed frame */ }
        }
      }
      finished = true;
    } catch (err) {
      if (err.name === "AbortError") {
        if (!answer) answer = "_Stopped._";
      } else {
        answer = answer || `**Error:** ${err.message}\n\nIs OVMS running and a model loaded? Open **Models** to check.`;
        toast(err.message, true);
      }
    } finally {
      state.streaming = false;
      state.controller = null;
      $("#send").disabled = false;
      $("#stop").classList.add("hidden");
      $("#chat-status").textContent = "";
      setAssistantContent(assistant, answer || "_No response._");
      const badges = [];
      if (memoryCount > 0 && requestId) {
        badges.push(`<button class="badge button" data-inspect="${escapeAttr(requestId)}">${memoryCount} memories used</button>`);
      } else if (finished) {
        badges.push('<span class="badge">no memory used</span>');
      }
      metaEl.innerHTML = badges.join("") +
        `<span class="muted small">${escapeHtml(fmtDate(Date.now()))}</span>`;
      await refreshConversations();
      if (finished) refreshState();
    }
  }

  function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = Math.min(220, el.scrollHeight) + "px";
  }

  /* ------------------------------------------------------------ documents in chat */
  function renderDocumentJob(job) {
    const box = $("#chat-doc-job");
    if (!box) return;
    box.classList.remove("hidden");
    $("#chat-doc-name").textContent = job.filename || "document.pdf";
    const total = Number(job.pages_total || 0);
    const done = Number(job.pages_done || 0);
    const pct = total ? Math.max(4, Math.min(100, Math.round((done / total) * 100))) : 4;
    $("#chat-doc-bar").style.width = `${pct}%`;
    const stateText = {
      queued: "Queued…",
      running: `Reading pages ${done}/${total || "?"} — batch ${job.batches_done || 0}/${job.batches_total || "?"}`,
      done: "Saved to memory.",
      error: `Failed: ${job.error || "unknown error"}`,
      cancelled: "Cancelled.",
    }[job.state] || job.state;
    $("#chat-doc-status").textContent = stateText;
  }

  function stopDocumentPolling() {
    if (state.chatDocTimer) {
      clearInterval(state.chatDocTimer);
      state.chatDocTimer = null;
    }
  }

  function appendDocumentMemory(job) {
    if (!job || !job.final) return;
    const text = `**Document memory: ${job.title || job.filename}**\n\n${job.final}\n\n` +
      `_Saved as a durable memory entry (${job.pages_total} pages). Ask about it in any chat._`;
    appendMessage("assistant", text);
  }

  async function pollDocumentJob() {
    const job = state.chatDocJob;
    if (!job || !job.id) { stopDocumentPolling(); return; }
    try {
      const current = await api(`/documents/jobs/${encodeURIComponent(job.id)}`);
      state.chatDocJob = current;
      renderDocumentJob(current);
      if (["done", "error", "cancelled"].includes(current.state)) {
        stopDocumentPolling();
        if (current.state === "done") {
          appendDocumentMemory(current);
          toast("Document analysis saved to memory.");
          refreshState();
        } else if (current.state === "error") {
          toast(current.error || "Document analysis failed.", true);
        }
      }
    } catch (err) {
      stopDocumentPolling();
      toast(err.message, true);
    }
  }

  async function startDocumentAnalysis(file) {
    if (!file) return;
    if (!/\.pdf$/i.test(file.name || "")) {
      toast("Choose a PDF file.", true);
      return;
    }
    stopDocumentPolling();
    state.chatDocJob = { id: "", filename: file.name, state: "queued", pages_total: 0 };
    renderDocumentJob(state.chatDocJob);
    try {
      const res = await fetch(
        `/api/v1/documents/analyze?filename=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          headers: { "X-AIO-Token": TOKEN, "Content-Type": "application/octet-stream" },
          body: file,
        }
      );
      let data = null;
      try { data = await res.json(); } catch { data = null; }
      if (!res.ok) {
        throw new Error((data && data.error && data.error.message) || `Upload failed (HTTP ${res.status})`);
      }
      state.chatDocJob = data.job;
      renderDocumentJob(data.job);
      state.chatDocTimer = setInterval(pollDocumentJob, 1500);
    } catch (err) {
      $("#chat-doc-status").textContent = err.message;
      toast(err.message, true);
    } finally {
      $("#chat-document-input").value = "";
    }
  }

  /* ------------------------------------------------------------ images */
  async function refreshGallery() {
    try {
      const payload = await api("/images");
      state.gallery = payload.data || [];
      $("#gallery-count").textContent = `${state.gallery.length} image${state.gallery.length === 1 ? "" : "s"}`;
      const gallery = $("#gallery");
      if (!state.gallery.length) {
        gallery.innerHTML = '<p class="muted small">No images yet. Generate one above.</p>';
        return;
      }
      gallery.innerHTML = state.gallery.map((img) => `
        <div class="gallery-item ${state.selectedImage === img.id ? "selected" : ""}" data-image="${escapeAttr(img.id)}">
          <img src="${escapeAttr(img.url)}?token=${encodeURIComponent(TOKEN)}" alt="${escapeAttr(img.prompt || "generated image")}" loading="lazy">
          <div class="gallery-cap">${escapeHtml(img.prompt || "untitled")}</div>
        </div>`).join("");
    } catch (err) {
      toast(err.message, true);
    }
  }

  function selectImage(id) {
    state.selectedImage = id;
    const item = state.gallery.find((i) => i.id === id);
    const box = $("#img-selected");
    box.classList.remove("empty");
    if (item) {
      box.innerHTML = `<img src="${escapeAttr(item.url)}?token=${encodeURIComponent(TOKEN)}" alt="selected image">`;
    }
    $$(".gallery-item").forEach((el) => el.classList.toggle("selected", el.dataset.image === id));
  }

  async function generateImage() {
    const prompt = $("#img-prompt").value.trim();
    if (!prompt) { toast("Write a prompt first.", true); return; }
    const button = $("#img-generate");
    button.disabled = true;
    $("#img-status").textContent = "Generating… this can take a minute on the iGPU.";
    try {
      const seedRaw = $("#img-seed").value.trim();
      const payload = {
        model: ($("#img-model") && $("#img-model").value) || "sd15",
        prompt,
        negative_prompt: $("#img-negative").value.trim(),
        width: parseInt($("#img-width").value, 10) || 512,
        height: parseInt($("#img-height").value, 10) || 512,
        num_inference_steps: parseInt($("#img-steps").value, 10) || 20,
        n: parseInt($("#img-count").value, 10) || 1,
      };
      if (seedRaw) payload.seed = parseInt(seedRaw, 10);
      const result = await api("/images/generate", { method: "POST", body: payload });
      const count = (result.data || []).length;
      $("#img-status").textContent = `Generated ${count} image${count === 1 ? "" : "s"}.`;
      await refreshGallery();
      if (result.data && result.data[0]) selectImage(result.data[0].id);
    } catch (err) {
      $("#img-status").textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function editImage() {
    if (!state.selectedImage) { toast("Select an image from the gallery first.", true); return; }
    const prompt = $("#img-edit-prompt").value.trim();
    if (!prompt) { toast("Describe the edit first.", true); return; }
    const button = $("#img-edit");
    button.disabled = true;
    $("#img-edit-status").textContent = "Editing…";
    try {
      await api("/images/edit", {
        method: "POST",
        body: {
          image_id: state.selectedImage,
          prompt,
          model: ($("#img-edit-model") && $("#img-edit-model").value) || "sd15-inpaint",
        },
      });
      $("#img-edit-status").textContent = "Done.";
      await refreshGallery();
    } catch (err) {
      $("#img-edit-status").textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  /* ------------------------------------------------------------ memory */
  function renderMemoryStats() {
    const s = state.stats || {};
    $("#stat-messages").textContent = s.messages ?? "0";
    $("#stat-summaries").textContent = s.summaries ?? "0";
    $("#stat-topics").textContent = s.topics ?? "0";
    $("#stat-jobs").textContent = s.jobs_pending ?? "0";
  }

  async function refreshMemory() {
    renderMemoryStats();
    try {
      const [topics, summaries, jobs] = await Promise.all([
        api("/memory/topics"),
        api("/memory/summaries?limit=60"),
        api("/memory/jobs?limit=40"),
      ]);

      const select = $("#mem-topic");
      const current = select.value;
      select.innerHTML = '<option value="">All topics</option>' +
        (topics.data || []).map((t) => `<option value="${escapeAttr(t.slug)}">${escapeHtml(t.slug)} (${t.count})</option>`).join("");
      select.value = current;

      renderSummaries(summaries.data || []);
      renderJobs(jobs.data || []);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function reembedMemory() {
    const status = $("#reembed-status");
    const button = $("#mem-reembed");
    if (button) button.disabled = true;
    if (status) status.textContent = "Queueing semantic re-embedding…";
    try {
      const result = await api("/memory/reembed?limit=5000", { method: "POST" });
      if (status) status.textContent = `Queued re-embedding with ${result.model}. Watch Background jobs.`;
      toast("Semantic re-embedding queued.");
      setTimeout(() => { if (state.view === "memory") refreshMemory(); }, 1200);
    } catch (err) {
      if (status) status.textContent = "";
      toast(err.message, true);
    } finally {
      if (button) button.disabled = false;
    }
  }

  function renderSummaries(rows) {
    const box = $("#summary-list");
    if (!rows.length) {
      box.innerHTML = '<p class="muted small">No summaries yet. They are created after a reply, in the background.</p>';
      return;
    }
    box.innerHTML = rows.map((s) => `
      <div class="summary-item" data-summary="${escapeAttr(s.id)}">
        <div class="summary-title">${escapeHtml(s.title)}</div>
        <div class="summary-sub">${escapeHtml(s.topic_slug || "unsorted")} · salience ${Number(s.salience || 0).toFixed(2)} · ${s.message_count || 0} messages</div>
        <div class="summary-body">${escapeHtml(String(s.body || "").slice(0, 240))}${String(s.body || "").length > 240 ? "…" : ""}</div>
      </div>`).join("");
  }

  function renderJobs(rows) {
    const box = $("#job-list");
    if (!rows.length) { box.innerHTML = '<p class="muted small">No jobs yet.</p>'; return; }
    box.innerHTML = rows.map((j) => `
      <div class="job">
        <span class="state ${escapeAttr(j.state)}">${escapeHtml(j.state)}</span>
        <span>${escapeHtml(j.kind)}</span>
        <span class="muted small grow">${escapeHtml(j.last_error || "")}</span>
        <span class="muted small">${escapeHtml(fmtDate(j.created_at))}</span>
      </div>`).join("");
  }

  async function searchMemory() {
    const query = $("#mem-query").value.trim();
    if (!query) return;
    const box = $("#mem-results");
    box.innerHTML = '<p class="muted small">Searching…</p>';
    try {
      const payload = await api("/memory/search", { method: "POST", body: { query } });
      const rows = payload.data || [];
      if (!rows.length) {
        box.innerHTML = `<p class="muted small">Nothing relevant found (${payload.intent}, ${payload.latency_ms} ms).</p>`;
        return;
      }
      box.innerHTML = `<p class="muted small">${rows.length} result${rows.length === 1 ? "" : "s"} in ${payload.latency_ms} ms.</p>` +
        rows.map((r) => `
          <div class="result">
            <div class="result-head">
              <span class="badge">${escapeHtml(r.kind)}</span>
              <strong>${escapeHtml(r.title)}</strong>
              <span class="muted small grow">score ${Number(r.score || 0).toFixed(4)}${r.topic ? " · " + escapeHtml(r.topic) : ""}</span>
            </div>
            <div class="result-body">${escapeHtml(String(r.body || "").slice(0, 500))}</div>
          </div>`).join("");
    } catch (err) {
      box.innerHTML = "";
      toast(err.message, true);
    }
  }

  async function openSummary(id) {
    try {
      const payload = await api(`/memory/summaries/${encodeURIComponent(id)}`);
      const s = payload.summary || {};
      const messages = payload.messages || [];
      openModal(`
        <h2>${escapeHtml(s.title || "Summary")}</h2>
        <p class="muted small">${escapeHtml(s.topic_slug || "unsorted")} · salience ${Number(s.salience || 0).toFixed(2)} · ${escapeHtml(s.source || "")}</p>
        <div class="msg-content">${renderMarkdown(s.body || "")}</div>
        <h2 style="margin-top:18px">Linked messages</h2>
        ${messages.map((m) => `
          <div class="result">
            <div class="result-head"><span class="badge">${escapeHtml(m.role)}</span><span class="muted small">${escapeHtml(fmtDate(m.created_at))}</span></div>
            <div class="result-body">${escapeHtml(String(m.content || "").slice(0, 600))}</div>
          </div>`).join("") || '<p class="muted small">No linked messages.</p>'}
      `);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function openInspector(requestId) {
    try {
      const payload = await api(`/inspector/${encodeURIComponent(requestId)}`);
      const retrieval = payload.retrieval || {};
      const plan = retrieval.plan || {};
      const assembly = plan.assembly || {};
      const total = Math.max(1, assembly.context_tokens || 1);
      const rows = [
        ["system", assembly.system_tokens || 0, ""],
        ["current input", assembly.input_tokens || 0, ""],
        ["tool chain", assembly.tool_tokens || 0, ""],
        ["recent history", assembly.history_tokens || 0, ""],
        ["summaries", assembly.summary_tokens || 0, "warn"],
        ["exact excerpts", assembly.exact_tokens || 0, "danger"],
        ["output reserve", assembly.output_reserve || 0, ""],
      ];
      const bars = rows.map(([label, value, cls]) => {
        const pct = Math.min(100, (Number(value) / total) * 100);
        return `<div class="bar-row"><span>${escapeHtml(label)}</span>
          <span class="bar"><i class="${cls}" style="width:${pct.toFixed(1)}%"></i></span>
          <span>${Number(value)} tok</span></div>`;
      }).join("");

      const items = plan.selected || (plan.retrieval && plan.retrieval.selected) || [];
      const itemRows = items.map((i) => `
        <div class="result">
          <div class="result-head"><span class="badge">${escapeHtml(i.kind)}</span>
            <strong>${escapeHtml(i.title)}</strong>
            <span class="muted small grow">${Number(i.score || 0).toFixed(4)}</span></div>
          <div class="result-body">${escapeHtml(String(i.body || "").slice(0, 420))}</div>
        </div>`).join("") || '<p class="muted small">No memory items were selected for this reply.</p>';

      const trimmed = (assembly.trims || []).length;
      openModal(`
        <h2>What the model saw</h2>
        <p class="muted small">request ${escapeHtml(requestId)} · retrieval ${retrieval.latency_ms || 0} ms${trimmed ? " · trimmed: " + escapeHtml((assembly.trims || []).join(", ")) : ""}</p>
        <div class="bars">${bars}</div>
        <h2 style="margin-top:16px">Retrieved memory</h2>
        ${itemRows}
        <details style="margin-top:14px"><summary class="muted small">Raw plan</summary>
          <pre class="json">${escapeHtml(JSON.stringify(plan, null, 2))}</pre>
        </details>
      `);
    } catch (err) {
      toast(err.message, true);
    }
  }

  /* ------------------------------------------------------------ models */
  function fmtBytes(value) {
    const n = Number(value || 0);
    if (!n) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `${v.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
  }

  function renderModels() {
    const data = state.models;
    if (!data) return;
    const ovmsState = data.ovms_state || (data.ovms_running ? "ready" : "stopped");
    const stateLabel = {
      stopped: "Stopped",
      starting: "Starting",
      loading: "Loading models",
      ready: "Ready",
      error: "Failed to start",
    }[ovmsState] || ovmsState;

    $("#ovms-detail").textContent = ovmsState === "ready"
      ? `Running (pid ${data.ovms_pid}) on port ${data.ovms_port} · config ${data.config_path}`
      : ovmsState === "stopped"
        ? `Not running · config ${data.config_path}`
        : `${stateLabel}…`;

    const exe = data.ovms_executable || "";
    $("#ovms-hint").textContent = exe
      ? `Executable: ${exe}`
      : "OVMS not found. Install it or set the ovms.exe path in Settings.";

    const card = $("#ovms-progress-card");
    const busy = ovmsState === "starting" || ovmsState === "loading" || ovmsState === "error";
    card.classList.toggle("hidden", !busy);
    if (busy) {
      $("#ovms-stage").textContent = data.ovms_stage || stateLabel;
      $("#ovms-elapsed").textContent = data.ovms_elapsed_s
        ? `${Math.round(Number(data.ovms_elapsed_s))} s elapsed`
        : "";
      const bar = $("#ovms-bar");
      bar.classList.toggle("indeterminate", ovmsState !== "error");
      bar.classList.toggle("failed", ovmsState === "error");
      const err = $("#ovms-error");
      err.classList.toggle("hidden", !data.ovms_error);
      err.textContent = data.ovms_error || "";
      const logWrap = $("#ovms-log-wrap");
      const log = (data.ovms_log_tail || []).join("\n");
      logWrap.classList.toggle("hidden", !log);
      $("#ovms-log").textContent = log;
    }

    $("#ovms-start").disabled = ovmsState === "starting" || ovmsState === "loading" || ovmsState === "ready";
    $("#ovms-stop").disabled = ovmsState === "stopped" || ovmsState === "error";

    const list = $("#model-list");
    list.innerHTML = (data.models || []).map((m) => {
      const tags = [
        m.installed ? '<span class="tag ok">installed</span>' : '<span class="tag">not downloaded</span>',
        m.loaded ? '<span class="tag ok">loaded</span>' : "",
        `<span class="tag">${escapeHtml(m.license || "")}</span>`,
        `<span class="tag">~${escapeHtml(String(m.size_gb))} GB disk</span>`,
        m.ram_gb ? `<span class="tag">~${escapeHtml(String(m.ram_gb))} GB RAM</span>` : "",
      ].filter(Boolean).join("");

      const options = (m.devices || []).map((d) =>
        `<option value="${escapeAttr(d)}" ${d === m.device ? "selected" : ""}>${escapeHtml(d)}</option>`).join("");

      let actions = "";
      let progress = "";
      if (m.pulling) {
        const pct = m.pull_percent == null ? null : Math.max(0, Math.min(99, Number(m.pull_percent)));
        const downloaded = fmtBytes(m.pull_downloaded);
        const total = fmtBytes(m.pull_total);
        progress = `
          <div class="progress-head">
            <span>Downloading ${pct == null ? "" : `${pct.toFixed(0)}%`}</span>
            <span class="muted small">${downloaded} / ${total}</span>
          </div>
          <div class="progress"><i style="width:${pct == null ? 8 : Math.max(2, pct)}%"></i></div>`;
        actions = '<span class="muted small">Download in progress…</span>';
      } else if (!m.installed) {
        actions = `<button class="btn ghost small" data-pull="${escapeAttr(m.id)}">Download</button>`;
      } else if (m.loaded) {
        actions = `<select data-device="${escapeAttr(m.id)}">${options}</select>
          <button class="btn ghost small" data-load="${escapeAttr(m.id)}">Reload</button>
          <button class="btn danger small" data-unload="${escapeAttr(m.id)}">Unload</button>`;
      } else {
        actions = `<select data-device="${escapeAttr(m.id)}">${options}</select>
          <button class="btn primary small" data-load="${escapeAttr(m.id)}">Load</button>`;
      }

      const err = m.pull_error ? `<div class="tag err">${escapeHtml(m.pull_error)}</div>` : "";
      return `
        <div class="model-card ${m.loaded ? "loaded" : ""}">
          <div class="model-head">
            <div>
              <div class="model-name">${escapeHtml(m.name)}</div>
              <div class="model-repo">${escapeHtml(m.repo)}</div>
            </div>
            <span class="tag">${escapeHtml(m.role)}</span>
          </div>
          <div class="model-desc">${escapeHtml(m.description)}</div>
          ${m.spec ? `<div class="model-spec">${escapeHtml(m.spec)}</div>` : ""}
          <div class="model-tags">${tags}</div>
          ${progress}
          ${err}
          <div class="model-actions">${actions}</div>
        </div>`;
    }).join("");

    ensureModelPolling(data);
  }

  function ensureModelPolling(data = state.models) {
    const active = !!data && (
      data.ovms_state === "starting" ||
      data.ovms_state === "loading" ||
      (data.models || []).some((m) => m.pulling)
    );
    if (active && !modelsTimer) {
      modelsTimer = setInterval(refreshModels, 1500);
    } else if (!active && modelsTimer) {
      clearInterval(modelsTimer);
      modelsTimer = null;
    }
  }

  async function refreshModels() {
    try {
      state.models = await api("/models");
      renderModels();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function loadModel(id) {
    const select = document.querySelector(`select[data-device="${CSS.escape(id)}"]`);
    const device = select ? select.value : "GPU";
    try {
      const result = await api(`/models/${encodeURIComponent(id)}/load`, { method: "POST", body: { device } });
      if (result.reload && !result.reload.ok) {
        toast(`Saved, but OVMS did not reload: ${result.reload.message}`, true);
      } else {
        toast(`${id} loaded on ${device}.`);
      }
      await refreshModels();
      refreshState();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function unloadModel(id) {
    try {
      const result = await api(`/models/${encodeURIComponent(id)}/unload`, { method: "POST" });
      if (result.reload && !result.reload.ok) {
        toast(`Unloaded, but OVMS did not reload: ${result.reload.message}`, true);
      } else {
        toast(`${id} unloaded.`);
      }
      await refreshModels();
      refreshState();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function pullModel(id) {
    try {
      await api(`/models/${encodeURIComponent(id)}/pull`, { method: "POST" });
      toast(`Download started for ${id}. Progress is shown on its card.`);
      await refreshModels();
    } catch (err) {
      toast(err.message, true);
    }
  }

  /* ------------------------------------------------------------ setup */
  const REQUIREMENT_STATES = {
    installed: ["Installed", "ok"],
    missing: ["Not installed", "err"],
    queued: ["Queued", "warn"],
    downloading: ["Downloading", "warn"],
    installing: ["Installing", "warn"],
    failed: ["Failed", "err"],
  };

  function requirementBadge(component) {
    const key = component.state || (component.installed ? "installed" : "missing");
    const [label, cls] = REQUIREMENT_STATES[key] || [key, ""];
    return `<span class="tag ${cls}">${escapeHtml(label)}</span>`;
  }

  function renderRequirementCard(component) {
    const installed = !!component.installed;
    const state = component.state || (installed ? "installed" : "missing");
    const busy = !!component.installing || ["queued", "downloading", "installing"].includes(state);
    const percent = Math.max(0, Math.min(100, Number(component.percent || 0)));
    const size = component.downloaded
      ? `${fmtBytes(component.downloaded)}${component.total ? " / " + fmtBytes(component.total) : ""}`
      : "";
    const message = component.message || (installed ? "Ready" : "Waiting to install");
    const detected = component.detected_version || component.version || "";
    return `
      <div class="setup-item ${installed ? "installed" : ""} ${state === "failed" ? "failed" : ""}" data-component="${escapeAttr(component.id)}">
        <div class="setup-head">
          <div>
            <div class="setup-name">${escapeHtml(component.name)}</div>
            <div class="setup-version">${escapeHtml(component.id)}${component.version ? " · " + escapeHtml(component.version) : ""}</div>
            <div class="setup-path">${escapeHtml(component.path || "")}</div>
          </div>
          ${requirementBadge(component)}
        </div>
        <div class="model-desc">${escapeHtml(component.description || "")}</div>
        <div class="progress-head">
          <span>${escapeHtml(message)}</span>
          <span class="muted small">${busy ? percent.toFixed(0) + "%" : escapeHtml(size || detected)}</span>
        </div>
        <div class="progress ${busy && percent <= 0 ? "indeterminate" : ""} ${state === "failed" ? "failed" : ""}">
          <i style="width:${installed && !busy ? 100 : percent}%"></i>
        </div>
        <div class="setup-actions">
          <button class="btn ${installed ? "ghost" : "primary"} small" type="button" data-install="${escapeAttr(component.id)}" ${busy ? "disabled" : ""}>
            ${installed ? "Reinstall" : "Install"}
          </button>
          <button class="btn ghost small" type="button" data-component-log="${escapeAttr(component.id)}">Log</button>
          <span class="muted small">${escapeHtml(installed ? "installed" : component.required ? "required" : "optional")}</span>
        </div>
        ${component.error ? `<div class="error-box">${escapeHtml(component.error)}</div>` : ""}
      </div>`;
  }

  function updateSetupPill(payload) {
    const pill = $("#setup-pill");
    if (!pill || !payload) return;
    const missing = Number(payload.missing || 0);
    const installing = Number(payload.installing || 0);
    pill.classList.remove("hidden", "online", "warn", "offline");
    if (installing) {
      pill.classList.add("warn");
      pill.innerHTML = `<span class="dot"></span> installing ${installing}`;
    } else if (payload.ready) {
      pill.classList.add("online");
      pill.innerHTML = '<span class="dot"></span> setup ready';
    } else {
      pill.classList.add("offline");
      pill.innerHTML = `<span class="dot"></span> ${missing} missing`;
    }
  }

  function ensureSetupPolling(payload) {
    const active = (payload.components || []).some((c) =>
      c.installing || ["queued", "downloading", "installing"].includes(c.state)
    );
    if (active && !state.setupTimer) {
      state.setupTimer = setInterval(refreshSetup, 1500);
    } else if (!active && state.setupTimer) {
      clearInterval(state.setupTimer);
      state.setupTimer = null;
    }
  }

  function renderSetup(payload) {
    state.requirements = payload;
    updateSetupPill(payload);
    const components = payload.components || [];
    $("#setup-detail").textContent = payload.ready
      ? "All required components are installed. Reinstall any component if it stops working."
      : `${payload.missing} required component${payload.missing === 1 ? "" : "s"} still missing. Install them here; no separate installer is needed.`;
    $("#setup-tags").innerHTML = [
      payload.ready ? '<span class="tag ok">setup complete</span>' : '<span class="tag err">setup incomplete</span>',
      payload.installing ? `<span class="tag warn">${payload.installing} installing</span>` : "",
      `<span class="tag">${components.filter((c) => c.installed).length}/${components.length} installed</span>`,
    ].filter(Boolean).join("");
    $("#setup-tools-dir").textContent = payload.tools_dir || "—";
    $("#setup-gimp-dir").textContent = payload.gimp_dir || "—";
    const box = $("#setup-components");
    box.innerHTML = components.length
      ? components.map(renderRequirementCard).join("")
      : '<p class="muted small">No components reported. Press Check again.</p>';
    ensureSetupPolling(payload);
  }

  async function refreshSetup(showError = true) {
    try {
      const payload = await api("/requirements");
      renderSetup(payload);
    } catch (err) {
      if (showError) toast(err.message, true);
    }
  }

  async function refreshSetupPill() {
    try {
      updateSetupPill(await api("/requirements"));
    } catch { /* the Setup page reports details */ }
  }

  async function installRequirement(id) {
    try {
      await api(`/requirements/${encodeURIComponent(id)}/install`, { method: "POST" });
      toast(`Installing ${id}… progress is shown on its card.`);
      await refreshSetup();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function installAllRequirements() {
    try {
      const result = await api("/requirements/install-all", { method: "POST" });
      const started = (result.started || []).length;
      toast(started ? `Installing ${started} missing component${started === 1 ? "" : "s"}…` : "Nothing is missing.");
      await refreshSetup();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function checkRequirements() {
    const button = $("#setup-check");
    button.disabled = true;
    try {
      const payload = await api("/requirements/check", { method: "POST" });
      renderSetup(payload);
      toast(payload.ready ? "All required components are installed." : `${payload.missing} still missing.`);
    } catch (err) {
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function openLogModal(name) {
    try {
      const payload = await api(`/logs/${encodeURIComponent(name)}?lines=500`);
      openModal(`
        <h2>${escapeHtml(name)} log</h2>
        <p class="muted small">${escapeHtml(payload.path || "")}</p>
        <pre class="json log-modal">${escapeHtml((payload.lines || []).join("\n") || "No lines yet.")}</pre>
      `);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function refreshLiveLog(showError = false) {
    const name = $("#log-select").value || "setup";
    try {
      const payload = await api(`/logs/${encodeURIComponent(name)}?lines=400`);
      $("#log-path").textContent = payload.path || "";
      const box = $("#log-view");
      box.textContent = (payload.lines || []).join("\n") || "No lines yet.";
      if ($("#log-follow").checked) box.scrollTop = box.scrollHeight;
    } catch (err) {
      if (showError) toast(err.message, true);
    }
  }

  function stopLogPolling() {
    if (state.logTimer) {
      clearInterval(state.logTimer);
      state.logTimer = null;
    }
  }

  function startLogPolling() {
    stopLogPolling();
    refreshLiveLog(true);
    state.logTimer = setInterval(() => refreshLiveLog(false), 2500);
  }

  /* ------------------------------------------------------------ documents */
  function renderKnowledgeStats(stats) {
    const s = stats || {};
    $("#kb-stat-collections").textContent = s.collections ?? "0";
    $("#kb-stat-documents").textContent = s.documents ?? "0";
    $("#kb-stat-chunks").textContent = s.chunks ?? "0";
    $("#kb-stat-failed").textContent = s.failed ?? "0";
  }

  function renderKnowledgeDocs() {
    const box = $("#kb-doc-list");
    const filter = $("#kb-filter").value;
    const rows = (state.knowledge && state.knowledge.documents) || [];
    const shown = filter ? rows.filter((d) => d.collection === filter) : rows;
    if (!shown.length) {
      box.innerHTML = '<p class="muted small">No documents yet. Upload one or drop files in the inbox and scan.</p>';
      return;
    }
    box.innerHTML = shown.map((d) => `
      <div class="doc-row" data-kb-doc="${escapeAttr(d.id)}">
        <div class="doc-main">
          <div class="doc-title">${escapeHtml(d.title || d.source_path || "document")}</div>
          <div class="doc-sub">${escapeHtml(d.collection || "general")} · ${escapeHtml(d.kind || "")} ·
            ${Number(d.chunk_count || 0)} chunks · ${Number(d.tokens || 0)} tokens
            ${d.status && d.status !== "ready" ? ` · <span class="tag err">${escapeHtml(d.status)}</span>` : ""}</div>
          ${d.error ? `<div class="tag err">${escapeHtml(d.error)}</div>` : ""}
        </div>
        <button class="btn danger small" data-kb-delete="${escapeAttr(d.id)}" type="button">Delete</button>
      </div>`).join("");
  }

  function populateKnowledgeSelect(collections) {
    const select = $("#chat-knowledge");
    if (!select) return;
    const current = select.value;
    select.innerHTML = '<option value="">Knowledge: off</option>' +
      (collections || []).map((c) =>
        `<option value="${escapeAttr(c.slug)}">${escapeHtml(c.name || c.slug)} (${c.documents || 0})</option>`
      ).join("");
    if (current && (collections || []).some((c) => c.slug === current)) select.value = current;
  }

  async function refreshKnowledge() {
    try {
      const [status, documents] = await Promise.all([
        api("/knowledge/status"),
        api("/knowledge/documents?limit=300"),
      ]);
      state.knowledge = { status, documents: documents.data || [] };
      renderKnowledgeStats(status.stats);
      $("#kb-inbox").textContent = status.inbox || "—";
      populateKnowledgeSelect(status.collections || []);

      const filter = $("#kb-filter");
      const current = filter.value;
      filter.innerHTML = '<option value="">All collections</option>' +
        (status.collections || []).map((c) =>
          `<option value="${escapeAttr(c.slug)}">${escapeHtml(c.name || c.slug)} (${c.documents})</option>`).join("");
      filter.value = current;
      renderKnowledgeDocs();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function uploadKnowledge() {
    const input = $("#kb-file");
    const file = input.files && input.files[0];
    if (!file) { toast("Choose a file first.", true); return; }
    const collection = $("#kb-collection").value.trim() || "general";
    const status = $("#kb-status");
    const button = $("#kb-upload");
    button.disabled = true;
    status.textContent = `Uploading ${file.name}…`;
    try {
      const res = await fetch(
        `/api/v1/knowledge/upload?collection=${encodeURIComponent(collection)}&filename=${encodeURIComponent(file.name)}`,
        {
          method: "POST",
          headers: { "X-AIO-Token": TOKEN, "Content-Type": "application/octet-stream" },
          body: file,
        }
      );
      let data = null;
      try { data = await res.json(); } catch { data = null; }
      if (!res.ok) {
        throw new Error((data && data.error && data.error.message) || `Upload failed (HTTP ${res.status})`);
      }
      status.textContent = `Imported ${file.name}.`;
      input.value = "";
      toast("Document imported.");
      await refreshKnowledge();
    } catch (err) {
      status.textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function scanKnowledgeInbox() {
    const button = $("#kb-scan");
    button.disabled = true;
    $("#kb-status").textContent = "Scanning the inbox…";
    try {
      const result = await api("/knowledge/scan", { method: "POST" });
      $("#kb-status").textContent = `Added ${result.added}, skipped ${result.skipped}, failed ${(result.failed || []).length}.`;
      await refreshKnowledge();
    } catch (err) {
      $("#kb-status").textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function searchKnowledge() {
    const query = $("#kb-query").value.trim();
    if (!query) return;
    const box = $("#kb-results");
    box.innerHTML = '<p class="muted small">Searching…</p>';
    try {
      const payload = await api("/knowledge/search", { method: "POST", body: { query } });
      const rows = payload.data || [];
      if (!rows.length) {
        box.innerHTML = '<p class="muted small">No matching chunks.</p>';
        return;
      }
      box.innerHTML = `<p class="muted small">${rows.length} result${rows.length === 1 ? "" : "s"}.</p>` +
        rows.map((r) => `
          <div class="result">
            <div class="result-head">
              <span class="badge">${escapeHtml(r.collection || "general")}</span>
              <strong>${escapeHtml(r.title || "chunk")}</strong>
              <span class="muted small grow">score ${Number(r.score || 0).toFixed(4)}</span>
            </div>
            <div class="result-body">${escapeHtml(String(r.text || "").slice(0, 600))}</div>
          </div>`).join("");
    } catch (err) {
      box.innerHTML = "";
      toast(err.message, true);
    }
  }

  async function openKnowledgeDocument(id) {
    try {
      const payload = await api(`/knowledge/documents/${encodeURIComponent(id)}`);
      const doc = payload.document || {};
      const chunks = payload.chunks || [];
      openModal(`
        <h2>${escapeHtml(doc.title || "Document")}</h2>
        <p class="muted small">${escapeHtml(doc.collection || "general")} · ${escapeHtml(doc.kind || "")} ·
          ${Number(doc.chunk_count || chunks.length)} chunks · ${Number(doc.tokens || 0)} tokens</p>
        <p class="muted small">${escapeHtml(doc.source_path || "")}</p>
        ${doc.summary ? `<div class="msg-content">${renderMarkdown(doc.summary)}</div>` : ""}
        <h2 style="margin-top:16px">Chunks</h2>
        ${chunks.map((c) => `
          <div class="result">
            <div class="result-head"><span class="badge">#${Number(c.ordinal || 0) + 1}</span>
              <span class="muted small">${Number(c.tokens || 0)} tokens</span></div>
            <div class="result-body">${escapeHtml(String(c.text || "").slice(0, 900))}</div>
          </div>`).join("") || '<p class="muted small">No chunks.</p>'}
      `);
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function deleteKnowledgeDocument(id) {
    if (!window.confirm("Delete this document from the knowledge base?")) return;
    try {
      await api(`/knowledge/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast("Document deleted.");
      await refreshKnowledge();
    } catch (err) {
      toast(err.message, true);
    }
  }

  /* ------------------------------------------------------------ code */
  function renderCode(status) {
    const tags = [
      status.installed ? '<span class="tag ok">OpenCode found</span>' : '<span class="tag err">OpenCode not found</span>',
      status.running ? '<span class="tag ok">server running</span>' : '<span class="tag">server stopped</span>',
      status.isolated ? '<span class="tag ok">isolated data</span>' : "",
      status.auth_ready ? '<span class="tag ok">auto sign-in ready</span>' : "",
      status.version ? `<span class="tag">${escapeHtml(status.version)}</span>` : "",
      status.node_version ? `<span class="tag">Node ${escapeHtml(status.node_version)}</span>` : "",
    ].filter(Boolean).join("");
    $("#code-tags").innerHTML = tags;
    $("#code-detail").textContent = status.running
      ? `Running (pid ${status.pid}) at ${status.url} · workspace ${status.workspace} · data ${status.data_dir}`
      : status.installed
        ? `Installed but not running · port ${status.port}`
        : "OpenCode was not found. Install it from Setup, or set its path in Settings.";
    $("#code-workspace").value = status.workspace || "";
    $("#code-config-path").textContent = status.config_path || "—";
    $("#code-log-path").textContent = status.log || "—";
    $("#code-start").disabled = !status.installed || status.running;
    $("#code-stop").disabled = !status.running;
  }

  function showCodeEmbedMessage(message, retry = true) {
    $("#code-embed-empty-text").textContent = message;
    $("#code-embed-empty").classList.remove("hidden");
    $("#code-embed-start").classList.toggle("hidden", !retry);
  }

  function resetCodeEmbed(message, retry = false) {
    state.codeEmbedUrl = "";
    const frame = $("#code-frame");
    frame.dataset.url = "";
    frame.src = "about:blank";
    frame.classList.add("hidden");
    $("#code-embed-state").textContent = "not connected";
    $("#code-embed-state").className = "tag";
    $("#code-embed-note").textContent = "";
    showCodeEmbedMessage(message || "OpenCode is not running.", retry);
  }

  async function ensureCodeEmbed(force = false) {
    if (state.codeEmbedBusy) return;
    if (!force && state.codeEmbedUrl) {
      $("#code-embed-empty").classList.add("hidden");
      $("#code-frame").classList.remove("hidden");
      return;
    }
    state.codeEmbedBusy = true;
    $("#code-embed-state").textContent = "starting…";
    $("#code-embed-state").className = "tag warn";
    showCodeEmbedMessage("Starting OpenCode and preparing its local sign-in…", false);
    try {
      const payload = await api("/code/embed", { method: "POST" });
      if (!payload.embed_url) throw new Error("OpenCode did not return an embed URL.");
      state.codeEmbedUrl = payload.embed_url;
      const frame = $("#code-frame");
      if (force || frame.dataset.url !== payload.embed_url) {
        frame.dataset.url = payload.embed_url;
        frame.src = payload.embed_url;
      }
      frame.classList.remove("hidden");
      $("#code-embed-empty").classList.add("hidden");
      $("#code-embed-state").textContent = `running · port ${payload.port}`;
      $("#code-embed-state").className = "tag ok";
      $("#code-embed-note").textContent =
        `Isolated data: ${payload.data_dir} · Workspace: ${payload.workspace}`;
    } catch (err) {
      resetCodeEmbed(err.message, true);
      toast(err.message, true);
    } finally {
      state.codeEmbedBusy = false;
    }
  }

  async function refreshCode(autoStart = true) {
    try {
      const status = await api("/code/status");
      renderCode(status);
      if (status.installed && (status.running || autoStart)) {
        if (!status.running) showCodeEmbedMessage("Starting OpenCode for this workspace…", false);
        await ensureCodeEmbed();
      } else if (!status.running) {
        resetCodeEmbed(
          status.installed ? "OpenCode is stopped. Start it again when you need it." : "OpenCode is not installed yet. Install it from Setup.",
          status.installed
        );
      }
      await refreshCodeFiles();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function refreshCodeFiles() {
    try {
      const payload = await api("/code/files");
      const files = payload.files || [];
      $("#code-file-list").innerHTML = files.length
        ? files.map((f) => `
            <button type="button" class="file-row" data-code-file="${escapeAttr(f.path)}">
              <span>${escapeHtml(f.path)}</span>
              <span class="muted small">${fmtBytes(f.size)}</span>
            </button>`).join("")
        : '<p class="muted small">The workspace is empty.</p>';
    } catch (err) {
      $("#code-file-list").innerHTML = `<p class="muted small">${escapeHtml(err.message)}</p>`;
    }
  }

  async function openCodeFile(path) {
    const preview = $("#code-file-preview");
    preview.textContent = "Loading…";
    try {
      const payload = await api(`/code/file?path=${encodeURIComponent(path)}`);
      preview.textContent = payload.text || "";
      if (payload.truncated) preview.textContent += "\n\n… truncated at 200 KB";
    } catch (err) {
      preview.textContent = err.message;
    }
  }

  async function setCodeWorkspace() {
    const value = $("#code-workspace").value.trim();
    if (!value) { toast("Enter a folder path.", true); return; }
    try {
      const result = await api("/code/workspace", { method: "POST", body: { path: value } });
      $("#code-ws-status").textContent = `Workspace set to ${result.workspace}.`;
      toast("Workspace saved.");
      await refreshCode();
    } catch (err) {
      $("#code-ws-status").textContent = "";
      toast(err.message, true);
    }
  }

  async function startCodeServer() {
    try {
      const result = await api("/code/start", { method: "POST" });
      toast(result.already_running ? "OpenCode is already running." : `OpenCode started (pid ${result.pid}).`);
      await refreshCode();
    } catch (err) { toast(err.message, true); }
  }

  async function stopCodeServer() {
    try {
      await api("/code/stop", { method: "POST" });
      resetCodeEmbed("OpenCode is stopped. Start it again when you need it.", true);
      toast("OpenCode stopped.");
      await refreshCode(false);
    } catch (err) { toast(err.message, true); }
  }

  async function writeCodeConfig() {
    try {
      const result = await api("/code/config", { method: "POST" });
      toast(`Provider config written to ${result.path}`);
    } catch (err) { toast(err.message, true); }
  }

  /* ------------------------------------------------------------ reverse engineering */
  function renderReverse(status) {
    $("#re-tags").innerHTML = [
      status.node_ok ? '<span class="tag ok">Node supported</span>' : '<span class="tag err">Node unsupported</span>',
      status.node_portable ? '<span class="tag ok">portable Node</span>' : (status.node_version ? '<span class="tag warn">system Node</span>' : ""),
      status.running ? '<span class="tag ok">REA running</span>' : '<span class="tag">REA stopped</span>',
      status.command ? `<span class="tag">${escapeHtml(status.command)}</span>` : "",
      status.tools_cached ? `<span class="tag">${status.tools_cached} tools</span>` : "",
    ].filter(Boolean).join("");
    $("#re-detail").textContent = status.running
      ? `REA is running · ${status.node_message}${status.node_path ? " · " + status.node_path : ""}`
      : status.command_error || `Not running · ${status.node_message}${status.node_path ? " · " + status.node_path : ""}`;
    $("#re-start").disabled = !!status.running || !!status.command_error || !status.node_ok;
    $("#re-stop").disabled = !status.running;
    if (!status.running) {
      $("#re-tool").innerHTML = '<option value="">Start REA to list tools</option>';
    }
  }

  function renderReverseTools(tools) {
    const list = $("#re-tools");
    const select = $("#re-tool");
    if (!tools.length) {
      list.innerHTML = '<p class="muted small">REA reported no tools.</p>';
      select.innerHTML = '<option value="">No tools</option>';
      return;
    }
    list.innerHTML = tools.map((t) => `
      <div class="result">
        <div class="result-head"><span class="badge">tool</span><strong>${escapeHtml(t.name || "")}</strong></div>
        <div class="result-body">${escapeHtml(t.description || "")}</div>
      </div>`).join("");
    select.innerHTML = tools.map((t) =>
      `<option value="${escapeAttr(t.name || "")}">${escapeHtml(t.name || "")}</option>`).join("");
  }

  async function refreshReverse() {
    try {
      const status = await api("/re/status");
      state.reverse = status;
      renderReverse(status);
      if (status.running) {
        try {
          const tools = await api("/re/tools");
          renderReverseTools(tools.data || []);
        } catch { /* tool discovery is reported on start */ }
      }
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function startReverse() {
    try {
      toast("Starting REA over MCP… this can take a while.");
      const result = await api("/re/start", { method: "POST" });
      toast(`REA started with ${result.tools || 0} tools.`);
      await refreshReverse();
    } catch (err) { toast(err.message, true); }
  }

  async function stopReverse() {
    try {
      await api("/re/stop", { method: "POST" });
      toast("REA stopped.");
      await refreshReverse();
    } catch (err) { toast(err.message, true); }
  }

  async function callReverseTool() {
    const name = $("#re-tool").value;
    if (!name) { toast("Start REA and choose a tool.", true); return; }
    let args = {};
    try {
      args = JSON.parse($("#re-args").value || "{}");
    } catch {
      toast("Arguments must be valid JSON.", true);
      return;
    }
    if (!args || typeof args !== "object" || Array.isArray(args)) {
      toast("Arguments must be a JSON object.", true);
      return;
    }
    $("#re-result").textContent = "Calling…";
    try {
      const result = await api("/re/call", { method: "POST", body: { name, arguments: args } });
      $("#re-result").textContent = JSON.stringify(result.result, null, 2);
    } catch (err) {
      $("#re-result").textContent = err.message;
      toast(err.message, true);
    }
  }

  /* ------------------------------------------------------------ GIMP */
  function renderGimp(status) {
    const version = status.version || "";
    const wrongVersion = !!version && !status.version_ok;
    $("#gimp-tags").innerHTML = [
      status.installed ? '<span class="tag ok">GIMP found</span>' : '<span class="tag err">GIMP not found</span>',
      status.running ? '<span class="tag ok">running</span>' : '<span class="tag">not running</span>',
      status.embedded ? '<span class="tag ok">docked</span>' : "",
      status.supported ? '<span class="tag">Windows docking</span>' : '<span class="tag">external window</span>',
      version ? `<span class="tag ${wrongVersion ? "warn" : "ok"}">GIMP ${escapeHtml(version)}</span>` : "",
      status.plugin_installed
        ? `<span class="tag ok">plugin ${escapeHtml(status.plugin_version || "")}</span>`
        : '<span class="tag err">plugin missing</span>',
      status.sd_weights ? '<span class="tag ok">SD weights found</span>' : '<span class="tag err">no SD weights</span>',
      status.plugin_server_ready ? '<span class="tag ok">plugin server listening</span>' : "",
    ].filter(Boolean).join("");
    $("#gimp-detail").textContent = status.running
      ? `GIMP running (pid ${status.pid})${status.embedded ? " and docked" : " in its own window"}${status.last_error ? " · " + status.last_error : ""}`
      : status.installed
        ? `GIMP is installed but not running.${wrongVersion ? " The Intel plugin requires GIMP 3.0.6." : ""}`
        : "GIMP was not found. Install it from Setup, or set its executable path in Settings.";
    $("#gimp-profile").textContent = status.personal_dir || "—";
    $("#gimp-models").textContent = status.models_dir || "—";
    $("#gimp-plugin").textContent = status.plugin_installed
      ? `installed${status.plugin_version ? " · " + status.plugin_version : ""}`
      : "not installed — open Setup";
    $("#gimp-launch").disabled = !status.installed || status.running;
    $("#gimp-release").disabled = !status.running && !status.embedded;
    const issues = status.issues || [];
    $("#gimp-issues").innerHTML = issues.length
      ? issues.map((issue) => `<div class="result"><div class="result-body">${escapeHtml(issue)}</div></div>`).join("")
      : '<p class="muted small">No GIMP problems detected.</p>';
  }

  async function refreshGimp() {
    try {
      const status = await api("/gimp/status");
      renderGimp(status);
      const images = await api("/images?limit=200");
      const select = $("#gimp-image");
      const current = select.value;
      const rows = images.data || [];
      select.innerHTML = rows.length
        ? rows.map((img) => `<option value="${escapeAttr(img.id)}">${escapeHtml(img.prompt || img.id)} · ${escapeHtml(fmtDate(img.created_at))}</option>`).join("")
        : '<option value="">No images yet</option>';
      if (current) select.value = current;
      await refreshPhotoGimp();
    } catch (err) {
      toast(err.message, true);
    }
  }

  async function launchGimp() {
    const imageId = $("#gimp-image").value || null;
    try {
      toast("Launching GIMP…");
      const body = { embed: !!(state.settings && state.settings.gimp_embed) };
      if (imageId) body.image_id = imageId;
      const result = await api("/gimp/launch", { method: "POST", body });
      toast(result.note || (result.opened ? `Opened ${result.opened} in GIMP.` : "GIMP launched."));
      await refreshGimp();
    } catch (err) { toast(err.message, true); }
  }

  async function openGimpImage() {
    const imageId = $("#gimp-image").value;
    if (!imageId) { toast("Generate an image first.", true); return; }
    try {
      await api("/gimp/open", { method: "POST", body: { image_id: imageId } });
      toast("Opened in GIMP.");
    } catch (err) { toast(err.message, true); }
  }

  async function releaseGimpDock() {
    try {
      await api("/gimp/release", { method: "POST" });
      toast("Dock released; GIMP is left running.");
      await refreshGimp();
    } catch (err) { toast(err.message, true); }
  }

  async function refreshPhotoGimp() {
    try {
      const status = await api("/gimp/photogimp");
      const tag = $("#photogimp-tag");
      tag.textContent = status.installed
        ? `installed ${status.version || ""}`.trim()
        : status.available ? "not installed" : "manual install";
      tag.className = `tag ${status.installed ? "ok" : ""}`;
      $("#photogimp-install").disabled = !status.available;
      $("#photogimp-remove").disabled = !status.installed;
      $("#photogimp-status").textContent = status.installed
        ? `${status.files_present}/${status.files} files present in ${status.profile}. Licence ${status.license}.`
        : status.note;
    } catch (err) {
      $("#photogimp-tag").textContent = "unavailable";
      $("#photogimp-status").textContent = err.message;
    }
  }

  async function installPhotoGimp() {
    const button = $("#photogimp-install");
    button.disabled = true;
    $("#photogimp-status").textContent = "Downloading the official PhotoGIMP release…";
    try {
      const result = await api("/gimp/photogimp/install", { method: "POST" });
      $("#photogimp-status").textContent =
        `Installed PhotoGIMP ${result.version} (${result.installed_files} files, ${result.backed_up} backed up).`;
      toast("PhotoGIMP installed into the private GIMP profile.");
      await refreshPhotoGimp();
    } catch (err) {
      $("#photogimp-status").textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function removePhotoGimp() {
    if (!window.confirm("Remove PhotoGIMP and restore the files it replaced?")) return;
    const button = $("#photogimp-remove");
    button.disabled = true;
    try {
      const result = await api("/gimp/photogimp/remove", { method: "POST" });
      toast(`Removed ${result.removed} files; restored ${result.restored}.`);
      await refreshPhotoGimp();
    } catch (err) {
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  async function repairGimp() {
    const button = $("#gimp-repair");
    button.disabled = true;
    $("#photogimp-status").textContent = "Repairing the plugin model list…";
    try {
      const result = await api("/gimp/repair", { method: "POST" });
      const removed = (result.config && result.config.removed) || [];
      $("#photogimp-status").textContent = removed.length
        ? `Repaired the model list: removed ${removed.length} invalid URL entr${removed.length === 1 ? "y" : "ies"}.`
        : "Plugin configuration is already valid.";
      await refreshGimp();
    } catch (err) {
      $("#photogimp-status").textContent = "";
      toast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  /* ------------------------------------------------------------ settings */
  function fillSettings() {
    const s = state.settings;
    if (!s) return;
    const set = (sel, value) => { const el = $(sel); if (el) el.value = value ?? ""; };
    const check = (sel, value) => { const el = $(sel); if (el) el.checked = !!value; };
    set("#set-upstream", s.upstream_base_url);
    set("#set-chat-model", s.chat_model);
    set("#set-ovms-path", s.ovms_path);
    set("#set-models-dir", s.models_dir);
    check("#set-ovms-auto", s.ovms_auto_start !== false);
    set("#set-context", s.context_tokens);
    set("#set-output", s.max_output_tokens);
    set("#set-history", s.short_history_turns);
    set("#set-embed", s.embed_dim);
    set("#set-embedding-backend", s.embedding_backend || "hash");
    set("#set-embedding-model", s.embedding_model);
    set("#set-filing-model", s.filing_model);
    set("#set-temp", s.temperature);
    check("#set-memory", s.memory_enabled);
    check("#set-hint", s.memory_hint);
    set("#set-docling", s.docling_path);
    set("#set-opencode-path", s.opencode_path);
    set("#set-opencode-port", s.opencode_port);
    set("#set-workspace", s.workspace_dir);
    set("#set-rea-command", s.rea_command);
    set("#set-rea-version", s.rea_version);
    set("#set-ghidra", s.ghidra_path);
    set("#set-gimp-path", s.gimp_path);
    check("#set-gimp-embed", s.gimp_embed);

    const p = state.paths || {};
    $("#path-data").textContent = p.data || "—";
    $("#path-db").textContent = p.database || "—";
    $("#path-images").textContent = p.images || "—";
    $("#path-logs").textContent = p.logs || "—";
  }

  async function saveSettings(event) {
    event.preventDefault();
    const payload = {
      upstream_base_url: $("#set-upstream").value.trim(),
      chat_model: $("#set-chat-model").value.trim(),
      ovms_path: $("#set-ovms-path").value.trim(),
      models_dir: $("#set-models-dir").value.trim(),
      ovms_auto_start: $("#set-ovms-auto").checked,
      context_tokens: parseInt($("#set-context").value, 10),
      max_output_tokens: parseInt($("#set-output").value, 10),
      short_history_turns: parseInt($("#set-history").value, 10),
      embed_dim: parseInt($("#set-embed").value, 10),
      embedding_backend: $("#set-embedding-backend").value,
      embedding_model: $("#set-embedding-model").value.trim(),
      filing_model: $("#set-filing-model").value.trim(),
      temperature: parseFloat($("#set-temp").value),
      memory_enabled: $("#set-memory").checked,
      memory_hint: $("#set-hint").checked,
      docling_path: $("#set-docling").value.trim(),
      opencode_path: $("#set-opencode-path").value.trim(),
      opencode_port: parseInt($("#set-opencode-port").value, 10),
      workspace_dir: $("#set-workspace").value.trim(),
      rea_command: $("#set-rea-command").value.trim(),
      rea_version: $("#set-rea-version").value.trim(),
      ghidra_path: $("#set-ghidra").value.trim(),
      gimp_path: $("#set-gimp-path").value.trim(),
      gimp_embed: $("#set-gimp-embed").checked,
    };
    try {
      const result = await api("/settings", { method: "PUT", body: payload });
      state.settings = result.settings;
      $("#settings-status").textContent = "Saved.";
      toast("Settings saved.");
      applyPills({ settings: result.settings, models: state.models || {} });
      refreshState();
    } catch (err) {
      $("#settings-status").textContent = "";
      toast(err.message, true);
    }
  }

  /* ------------------------------------------------------------ modal */
  function openModal(html) {
    $("#modal-body").innerHTML = html;
    typesetMath($("#modal-body"));
    $("#modal").classList.remove("hidden");
  }
  function closeModal() {
    $("#modal").classList.add("hidden");
    $("#modal-body").innerHTML = "";
  }

  /* ------------------------------------------------------------ events */
  function connectEvents() {
    if (!window.EventSource) return;
    const source = new EventSource(`/api/v1/events?token=${encodeURIComponent(TOKEN)}`);
    source.addEventListener("model_pull", (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.state === "done") toast(`${data.model} downloaded.`);
        if (data.state === "failed") toast(`Download failed: ${data.error || data.model}`, true);
        refreshModels();
      } catch { /* ignore */ }
    });
    source.addEventListener("job_done", () => {
      if (state.view === "memory") refreshMemory();
      refreshState();
    });
    source.addEventListener("job_failed", () => {
      if (state.view === "memory") refreshMemory();
    });
    source.addEventListener("image_done", () => {
      if (state.view === "images") refreshGallery();
    });
    source.addEventListener("model_loaded", refreshModels);
    source.addEventListener("model_unloaded", refreshModels);
    source.addEventListener("ovms_status", (event) => {
      if (state.view === "models") refreshModels();
      refreshState();
    });
    source.addEventListener("knowledge_document", () => {
      if (state.view === "documents") refreshKnowledge();
    });
    source.addEventListener("code_status", (event) => {
      if (state.view !== "code") return;
      let running = true;
      try {
        const data = JSON.parse(event.data);
        running = data.running !== false;
      } catch { /* keep the safe default */ }
      refreshCode(running);
    });
    source.addEventListener("re_status", () => {
      if (state.view === "re") refreshReverse();
    });
    source.addEventListener("gimp_status", () => {
      if (state.view === "gimp") refreshGimp();
    });
    source.addEventListener("requirement", () => {
      // Progress is polled while the Setup view is open; avoid one filesystem
      // scan per download chunk here.
      if (state.view === "setup" && !state.setupTimer) refreshSetup(false);
    });
    source.addEventListener("requirements_ready", () => {
      if (state.view === "setup") refreshSetup(false);
      refreshSetupPill();
    });
  }

  /* ------------------------------------------------------------ wiring */
  function wire() {
    $$(".nav-item").forEach((button) => {
      button.addEventListener("click", () => showView(button.dataset.view));
    });

    $("#conversation-select").addEventListener("change", (event) => {
      loadConversation(event.target.value || null);
    });
    $("#new-chat").addEventListener("click", newConversation);

    $("#composer").addEventListener("submit", (event) => {
      event.preventDefault();
      sendMessage();
    });
    $("#chat-attach").addEventListener("click", () => $("#chat-image-input").click());
    $("#chat-image-input").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) attachChatImage(file);
    });
    $("#chat-attachment-remove").addEventListener("click", clearChatImage);
    $("#chat-add-document").addEventListener("click", () => $("#chat-document-input").click());
    $("#chat-document-input").addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) startDocumentAnalysis(file);
    });
    $("#prompt").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
    $("#prompt").addEventListener("input", (event) => autoGrow(event.target));
    $("#stop").addEventListener("click", () => {
      if (state.controller) state.controller.abort();
    });

    $("#messages").addEventListener("click", (event) => {
      const suggest = event.target.closest("[data-suggest]");
      if (suggest) {
        $("#prompt").value = suggest.dataset.suggest;
        autoGrow($("#prompt"));
        $("#prompt").focus();
        return;
      }
      const inspect = event.target.closest("[data-inspect]");
      if (inspect) openInspector(inspect.dataset.inspect);
    });

    $("#img-generate").addEventListener("click", generateImage);
    $("#img-edit").addEventListener("click", editImage);
    $("#gallery").addEventListener("click", (event) => {
      const item = event.target.closest("[data-image]");
      if (item) selectImage(item.dataset.image);
    });

    $("#mem-search").addEventListener("click", searchMemory);
    $("#mem-query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); searchMemory(); }
    });
    $("#mem-topic").addEventListener("change", async (event) => {
      const topic = event.target.value;
      try {
        const payload = await api(`/memory/summaries?limit=60${topic ? `&topic=${encodeURIComponent(topic)}` : ""}`);
        renderSummaries(payload.data || []);
      } catch (err) { toast(err.message, true); }
    });
    $("#summary-list").addEventListener("click", (event) => {
      const item = event.target.closest("[data-summary]");
      if (item) openSummary(item.dataset.summary);
    });
    $("#mem-consolidate").addEventListener("click", async () => {
      try {
        await api("/memory/consolidate", { method: "POST" });
        toast("Consolidation queued.");
      } catch (err) { toast(err.message, true); }
    });
    $("#mem-reembed").addEventListener("click", reembedMemory);

    $("#model-list").addEventListener("click", (event) => {
      const pull = event.target.closest("[data-pull]");
      if (pull) { pullModel(pull.dataset.pull); return; }
      const load = event.target.closest("[data-load]");
      if (load) { loadModel(load.dataset.load); return; }
      const unload = event.target.closest("[data-unload]");
      if (unload) unloadModel(unload.dataset.unload);
    });
    $("#ovms-start").addEventListener("click", async () => {
      try {
        const r = await api("/ovms/start", { method: "POST" });
        toast(r.already_running
          ? "OVMS is already running."
          : `OVMS is starting (pid ${r.pid}) — waiting for the model server.`);
        refreshModels();
      } catch (err) { toast(err.message, true); }
    });
    $("#ovms-stop").addEventListener("click", async () => {
      try {
        await api("/ovms/stop", { method: "POST" });
        toast("OVMS stopped.");
        refreshModels();
      } catch (err) { toast(err.message, true); }
    });
    $("#ovms-reload").addEventListener("click", async () => {
      try {
        const r = await api("/ovms/reload", { method: "POST" });
        toast(r.ok ? "OVMS reloaded." : r.message, !r.ok);
        refreshModels();
      } catch (err) { toast(err.message, true); }
    });

    $("#kb-upload").addEventListener("click", uploadKnowledge);
    $("#kb-scan").addEventListener("click", scanKnowledgeInbox);
    $("#kb-search").addEventListener("click", searchKnowledge);
    $("#kb-query").addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); searchKnowledge(); }
    });
    $("#kb-filter").addEventListener("change", renderKnowledgeDocs);
    $("#kb-doc-list").addEventListener("click", (event) => {
      const del = event.target.closest("[data-kb-delete]");
      if (del) { deleteKnowledgeDocument(del.dataset.kbDelete); return; }
      const doc = event.target.closest("[data-kb-doc]");
      if (doc) openKnowledgeDocument(doc.dataset.kbDoc);
    });

    $("#code-start").addEventListener("click", startCodeServer);
    $("#code-stop").addEventListener("click", stopCodeServer);
    $("#code-config").addEventListener("click", writeCodeConfig);
    $("#code-set-workspace").addEventListener("click", setCodeWorkspace);
    $("#code-refresh").addEventListener("click", refreshCodeFiles);
    $("#code-file-list").addEventListener("click", (event) => {
      const row = event.target.closest("[data-code-file]");
      if (row) openCodeFile(row.dataset.codeFile);
    });
    $("#code-embed-start").addEventListener("click", () => ensureCodeEmbed(true));
    $("#code-embed-reload").addEventListener("click", () => {
      state.codeEmbedUrl = "";
      ensureCodeEmbed(true);
    });
    $("#code-embed-external").addEventListener("click", () => {
      if (!state.codeEmbedUrl) { toast("Start OpenCode first.", true); return; }
      window.open(state.codeEmbedUrl, "_blank", "noopener,noreferrer");
    });

    $("#setup-install-all").addEventListener("click", installAllRequirements);
    $("#setup-check").addEventListener("click", checkRequirements);
    $("#setup-components").addEventListener("click", (event) => {
      const install = event.target.closest("[data-install]");
      if (install) { installRequirement(install.dataset.install); return; }
      const log = event.target.closest("[data-component-log]");
      if (log) openLogModal(`setup-${log.dataset.componentLog}`);
    });
    $("#log-refresh").addEventListener("click", () => refreshLiveLog(true));
    $("#log-select").addEventListener("change", () => refreshLiveLog(true));
    document.addEventListener("click", (event) => {
      const button = event.target.closest("[data-open-log]");
      if (button) openLogModal(button.dataset.openLog);
    });

    $("#re-start").addEventListener("click", startReverse);
    $("#re-stop").addEventListener("click", stopReverse);
    $("#re-refresh").addEventListener("click", refreshReverse);
    $("#re-call").addEventListener("click", callReverseTool);

    $("#gimp-launch").addEventListener("click", launchGimp);
    $("#gimp-open").addEventListener("click", openGimpImage);
    $("#gimp-release").addEventListener("click", releaseGimpDock);
    $("#photogimp-install").addEventListener("click", installPhotoGimp);
    $("#photogimp-remove").addEventListener("click", removePhotoGimp);
    $("#gimp-repair").addEventListener("click", repairGimp);

    $("#settings-form").addEventListener("submit", saveSettings);

    $("#modal-close").addEventListener("click", closeModal);
    $("#modal").addEventListener("click", (event) => {
      if (event.target.id === "modal") closeModal();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeModal();
    });
  }

  async function init() {
    wire();
    try {
      const payload = await api("/state");
      state.settings = payload.settings;
      state.stats = payload.stats;
      state.models = payload.models;
      state.paths = payload.paths;
      applyPills(payload);
    } catch (err) {
      toast(err.message, true);
    }

    await refreshConversations();
    await refreshKnowledge();
    const first = state.conversations[0];
    if (first) {
      $("#conversation-select").value = first.id;
      await loadConversation(first.id);
    } else {
      showEmptyChat();
    }
    connectEvents();
    refreshSetupPill();
    setInterval(() => { if (!state.streaming) refreshState(); }, 30000);
    $("#prompt").focus();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
