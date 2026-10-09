var genres = document.querySelector("#genres");
var form = document.querySelector("#write");
var titleInput = document.querySelector("#title");
var temperatureInput = document.querySelector("#temperature");
var maxTokensInput = document.querySelector("#max-tokens");
var topKInput = document.querySelector("#top-k");
var topPInput = document.querySelector("#top-p");
var submit = document.querySelector("#submit");
var statusLine = document.querySelector("#status");
var statusText = document.querySelector("#status-text");
var result = document.querySelector("#result");
var poemTitle = document.querySelector("#poem-title");
var poemGenre = document.querySelector("#poem-genre");
var poem = document.querySelector("#poem");
var tokenCount = document.querySelector("#token-count");
var copyButton = document.querySelector("#copy");
var againButton = document.querySelector("#again");
var clearButton = document.querySelector("#clear");

// The API serves this page, so requests stay on the same origin (works locally and when hosted).
var API_BASE = "";
var currentGenre = document.documentElement.dataset.theme || "Love";
var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
var loadingTimer = 0;
var loadingIndex = 0;

titleInput.placeholder = THEMES[currentGenre].copy.placeholder;
Atmosphere.start(document.querySelector("#atmosphere"), currentGenre);

Object.keys(THEMES).forEach(function (name) {
  var theme = THEMES[name];
  var label = document.createElement("label");
  label.className = "genre";
  label.style.setProperty("--pill-font", theme.fonts.display);
  var input = document.createElement("input");
  input.type = "radio";
  input.name = "genre";
  input.value = name;
  input.checked = name === currentGenre;
  var face = document.createElement("span");
  face.className = "genre-face";
  var icon = document.createElement("span");
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = theme.emoji;
  face.append(icon, document.createTextNode(" " + name));
  label.append(input, face);
  input.addEventListener("change", function () {
    if (input.checked) {
      selectGenre(name);
    }
  });
  genres.append(label);
});

function selectGenre(name) {
  if (!THEMES[name] || name === currentGenre) {
    return;
  }
  ensureFonts(name);
  var finish = function () {
    currentGenre = name;
    applyTheme(name);
    Atmosphere.setTheme(name);
    titleInput.placeholder = THEMES[name].copy.placeholder;
    rememberGenre(name);
    document.body.classList.remove("is-fading");
  };
  if (reducedMotion) {
    finish();
    return;
  }
  document.body.classList.add("is-fading");
  window.setTimeout(finish, 200);
}

function ensureFonts(name) {
  if (loadedFontThemes[name]) {
    return;
  }
  var link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = THEMES[name].fontHref;
  document.head.appendChild(link);
  loadedFontThemes[name] = true;
}

form.addEventListener("submit", function (event) {
  event.preventDefault();
  var title = titleInput.value.trim();
  var genre = new FormData(form).get("genre");
  if (!title || !genre) {
    showStatus(THEMES[currentGenre].copy.error, true);
    return;
  }
  if (!form.reportValidity()) {
    return;
  }
  writePoem(title, genre);
});

againButton.addEventListener("click", function () {
  form.requestSubmit();
});

clearButton.addEventListener("click", function () {
  poem.textContent = "";
  tokenCount.textContent = "";
  copyButton.disabled = true;
  clearButton.disabled = true;
});

copyButton.addEventListener("click", async function () {
  var text = poem.textContent;
  if (!text) {
    return;
  }
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
  } catch (err) {
    copyButton.textContent = "Copy failed";
  }
  window.setTimeout(function () {
    copyButton.textContent = "Copy poem";
  }, 1600);
});

async function writePoem(title, genre) {
  submit.disabled = true;
  againButton.disabled = true;
  copyButton.disabled = true;
  clearButton.disabled = true;
  submit.textContent = "Writing…";
  startLoading(genre);

  try {
    var response = await fetch(API_BASE + "/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title,
        genre: genre,
        temperature: Number(temperatureInput.value),
        max_tokens: Number(maxTokensInput.value),
        top_k: optionalNumber(topKInput.value),
        top_p: optionalNumber(topPInput.value),
      }),
    });
    if (!response.ok) {
      showStatus(THEMES[genre].copy.error, true);
      return;
    }
    stopLoading();
    statusText.textContent = "";
    poemTitle.textContent = title;
    poemGenre.textContent = genre;
    poem.textContent = "";
    tokenCount.textContent = "";
    result.hidden = false;
    result.classList.remove("is-open");
    window.requestAnimationFrame(function () {
      result.classList.add("is-open");
    });
    var tokens = await readPoemStream(response);
    tokenCount.textContent = tokens === null ? "" : tokens + " tokens";
  } catch (err) {
    showStatus(THEMES[genre].copy.error, true);
  } finally {
    stopLoading();
    submit.disabled = false;
    againButton.disabled = false;
    copyButton.disabled = poem.textContent.length === 0;
    clearButton.disabled = poem.textContent.length === 0;
    submit.textContent = "Write";
  }
}

function startLoading(genre) {
  var messages = THEMES[genre].copy.loading;
  loadingIndex = 0;
  showStatus(messages[0], false);
  statusLine.classList.add("is-loading");
  window.clearInterval(loadingTimer);
  loadingTimer = window.setInterval(function () {
    loadingIndex = (loadingIndex + 1) % messages.length;
    statusText.textContent = messages[loadingIndex];
  }, 2500);
}

function stopLoading() {
  window.clearInterval(loadingTimer);
  statusLine.classList.remove("is-loading");
}

function showStatus(message, isError) {
  statusText.textContent = message;
  statusLine.classList.toggle("error", isError);
  if (isError) {
    statusLine.classList.remove("is-loading");
  }
}

function optionalNumber(value) {
  var raw = value.trim();
  if (raw === "") {
    return null;
  }
  return Number(raw);
}

async function readPoemStream(response) {
  var reader = response.body.getReader();
  var decoder = new TextDecoder();
  var buffer = "";
  var tokens = null;

  while (true) {
    var chunk = await reader.read();
    if (chunk.done) {
      break;
    }
    buffer += decoder.decode(chunk.value, { stream: true });
    var events = buffer.split("\n\n");
    buffer = events.pop() || "";
    events.forEach(function (event) {
      var data = eventData(event);
      if (!data) {
        return;
      }
      if (typeof data.poem === "string") {
        poem.textContent = data.poem;
      }
      if (data.done) {
        tokens = data.tokens;
      }
    });
  }
  return tokens;
}

function eventData(raw) {
  var lines = raw.split("\n");
  var line = lines.find(function (item) {
    return item.indexOf("data: ") === 0;
  });
  if (!line) {
    return null;
  }
  return JSON.parse(line.slice("data: ".length));
}
