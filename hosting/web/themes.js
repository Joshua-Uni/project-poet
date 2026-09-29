var loadedFontThemes = { Love: true };

var THEMES = {
  Love: {
    emoji: "♥",
    scheme: "light",
    fontHref: "https://fonts.googleapis.com/css2?family=Great+Vibes&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap",
    fonts: {
      display: '"Great Vibes", Palatino, "Palatino Linotype", Georgia, serif',
      poem: '"Libre Baskerville", Georgia, "Times New Roman", serif',
    },
    colors: {
      "--bg-1": "#fdeff2",
      "--bg-2": "#f6d5df",
      "--bg-3": "#efc2d0",
      "--surface": "#fff7f8",
      "--text": "#3d2433",
      "--muted": "#5c3a4a",
      "--accent": "#8e2f4f",
      "--on-accent": "#fff7f8",
      "--border": "#c9899c",
      "--glow": "#e7b0c2",
    },
    particles: { kind: "petals", count: 58 },
    copy: {
      placeholder: "A letter never sent",
      error: "Those words slipped away. Try again.",
      loading: [
        "Whispering to the stars",
        "Pressing petals between pages",
        "Warming the ink",
        "Listening for a heartbeat",
      ],
    },
  },
  Anger: {
    emoji: "🔥",
    scheme: "dark",
    fontHref: "https://fonts.googleapis.com/css2?family=Oswald:wght@500;600&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap",
    fonts: {
      display: '"Oswald", "Arial Narrow", Impact, sans-serif',
      poem: '"Source Serif 4", Georgia, "Times New Roman", serif',
    },
    colors: {
      "--bg-1": "#140a08",
      "--bg-2": "#3a1610",
      "--bg-3": "#5c2418",
      "--surface": "#2c1612",
      "--text": "#f8ece6",
      "--muted": "#f0d2c6",
      "--accent": "#ff7a45",
      "--on-accent": "#2a100c",
      "--border": "#a34a32",
      "--glow": "#ff8a5b",
    },
    particles: { kind: "embers", count: 64 },
    copy: {
      placeholder: "The door you slammed",
      error: "The fire went out. Strike again.",
      loading: [
        "Stoking the embers",
        "Sharpening the words",
        "Striking the page",
        "Letting sparks fly",
      ],
    },
  },
  Sadness: {
    emoji: "🌧",
    scheme: "light",
    fontHref: "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Literata:ital,wght@0,400;0,600;1,400&display=swap",
    fonts: {
      display: '"Cormorant Garamond", Palatino, Georgia, serif',
      poem: '"Literata", Georgia, "Times New Roman", serif',
    },
    colors: {
      "--bg-1": "#e7eef4",
      "--bg-2": "#d5e0ea",
      "--bg-3": "#c5d3e0",
      "--surface": "#f7f9fb",
      "--text": "#1c2833",
      "--muted": "#3d5164",
      "--accent": "#3d5a73",
      "--on-accent": "#f7f9fb",
      "--border": "#8aa0b4",
      "--glow": "#b7c9d8",
    },
    particles: { kind: "rain", count: 70 },
    copy: {
      placeholder: "The chair left empty",
      error: "That thought drifted off. Try again.",
      loading: [
        "Waiting for the rain",
        "Gathering quiet thoughts",
        "Holding a long pause",
        "Watching the light fade",
      ],
    },
  },
  Joy: {
    emoji: "☀",
    scheme: "light",
    fontHref: "https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&family=Nunito:ital,wght@0,400;0,700;1,400&display=swap",
    fonts: {
      display: '"Fredoka", "Trebuchet MS", "Segoe UI", sans-serif',
      poem: '"Nunito", "Segoe UI", "Trebuchet MS", sans-serif',
    },
    colors: {
      "--bg-1": "#fff6df",
      "--bg-2": "#ffe7b0",
      "--bg-3": "#ffd8c4",
      "--surface": "#fffdf8",
      "--text": "#2a2114",
      "--muted": "#5a4630",
      "--accent": "#8f4e00",
      "--on-accent": "#fffdf8",
      "--border": "#e0b15a",
      "--glow": "#ffd27a",
    },
    particles: { kind: "motes", count: 54 },
    copy: {
      placeholder: "Morning you woke laughing",
      error: "That spark fizzled. Try again.",
      loading: [
        "Catching a bright thought",
        "Shaking loose some laughter",
        "Opening the window wide",
        "Bottling a little sunlight",
      ],
    },
  },
  Fear: {
    emoji: "☾",
    scheme: "dark",
    fontHref: "https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap",
    fonts: {
      display: '"Cinzel", Palatino, "Palatino Linotype", serif',
      poem: '"Libre Baskerville", Georgia, "Times New Roman", serif',
    },
    colors: {
      "--bg-1": "#0c1016",
      "--bg-2": "#1a2333",
      "--bg-3": "#2c3a4e",
      "--surface": "#141a24",
      "--text": "#e8edf5",
      "--muted": "#d5deea",
      "--accent": "#a9bdd8",
      "--on-accent": "#10151e",
      "--border": "#5c7090",
      "--glow": "#8ea4c4",
    },
    particles: { kind: "stars", count: 62 },
    copy: {
      placeholder: "The light down the hall",
      error: "Something moved in the dark. Try again.",
      loading: [
        "Listening in the dark",
        "Checking the shadows",
        "Holding a small light",
        "Stepping past the door",
      ],
    },
  },
  Surprise: {
    emoji: "✶",
    scheme: "light",
    fontHref: "https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;1,500&family=Nunito+Sans:ital,wght@0,400;0,600;1,400&display=swap",
    fonts: {
      display: '"Fraunces", Georgia, "Times New Roman", serif',
      poem: '"Nunito Sans", "Segoe UI", "Trebuchet MS", sans-serif',
    },
    colors: {
      "--bg-1": "#f6e9ff",
      "--bg-2": "#ffe8c8",
      "--bg-3": "#f8d4ea",
      "--surface": "#fffaf6",
      "--text": "#2c2238",
      "--muted": "#574566",
      "--accent": "#7b3f96",
      "--on-accent": "#fffaf6",
      "--border": "#d7b0e4",
      "--glow": "#e7c4f2",
    },
    particles: { kind: "sparks", count: 52 },
    copy: {
      placeholder: "The gift you didn't expect",
      error: "That trick missed. Try again.",
      loading: [
        "Turning the page early",
        "Shaking the kaleidoscope",
        "Waiting for the twist",
        "Opening the wrong drawer",
      ],
    },
  },
};

function applyTheme(name) {
  var theme = THEMES[name] || THEMES.Love;
  var root = document.documentElement;
  root.dataset.theme = theme === THEMES[name] ? name : "Love";
  Object.keys(theme.colors).forEach(function (key) {
    root.style.setProperty(key, theme.colors[key]);
  });
  root.style.setProperty("--font-display", theme.fonts.display);
  root.style.setProperty("--font-poem", theme.fonts.poem);
  root.style.colorScheme = theme.scheme;
  Object.keys(THEMES).forEach(function (genre) {
    root.style.setProperty("--font-" + genre, THEMES[genre].fonts.display);
  });
  return theme;
}

function savedGenre() {
  try {
    var saved = localStorage.getItem("poet-genre");
    if (saved && THEMES[saved]) {
      return saved;
    }
  } catch (err) {
    // Storage can be blocked. The page still opens on Love.
  }
  return "Love";
}

function rememberGenre(name) {
  try {
    localStorage.setItem("poet-genre", name);
  } catch (err) {
    // Ignore a full or blocked store. The theme still applies for this visit.
  }
}
