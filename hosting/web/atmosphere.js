var Atmosphere = {
  canvas: null,
  ctx: null,
  particles: [],
  themeName: "Love",
  reduced: false,
  running: false,
  raf: 0,

  start: function (canvas, themeName) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.themeName = themeName;
    this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.resize();
    var self = this;
    window.addEventListener("resize", function () {
      self.resize();
    });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
        self.stop();
      } else if (!self.reduced) {
        self.loop();
      }
    });
    var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    motion.addEventListener("change", function () {
      self.reduced = motion.matches;
      if (self.reduced) {
        self.stop();
      } else {
        self.loop();
      }
    });
    if (!this.reduced) {
      this.loop();
    }
  },

  setTheme: function (name) {
    this.themeName = name;
    this.seed();
  },

  count: function () {
    var base = THEMES[this.themeName].particles.count;
    return window.innerWidth < 700 ? Math.round(base * 0.5) : base;
  },

  resize: function () {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = window.innerWidth;
    var height = window.innerHeight;
    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
    this.canvas.style.width = width + "px";
    this.canvas.style.height = height + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.seed();
  },

  seed: function () {
    var total = this.count();
    var width = window.innerWidth;
    var height = window.innerHeight;
    this.particles = [];
    for (var i = 0; i < total; i += 1) {
      this.particles.push(this.make(width, height, true));
    }
  },

  make: function (width, height, anywhere) {
    var kind = THEMES[this.themeName].particles.kind;
    return {
      x: Math.random() * width,
      y: anywhere ? Math.random() * height : (kind === "rain" || kind === "stars" ? -20 : height + 20),
      size: 2 + Math.random() * 5,
      speed: 0.25 + Math.random() * 0.9,
      drift: Math.random() * Math.PI * 2,
      spin: Math.random() * Math.PI * 2,
      variant: Math.random(),
    };
  },

  loop: function () {
    if (this.running || this.reduced || document.hidden) {
      return;
    }
    this.running = true;
    var self = this;
    var tick = function (now) {
      if (!self.running) {
        return;
      }
      self.draw(now || 0);
      self.raf = window.requestAnimationFrame(tick);
    };
    this.raf = window.requestAnimationFrame(tick);
  },

  stop: function () {
    this.running = false;
    if (this.raf) {
      window.cancelAnimationFrame(this.raf);
    }
  },

  draw: function (now) {
    var ctx = this.ctx;
    var width = window.innerWidth;
    var height = window.innerHeight;
    var kind = THEMES[this.themeName].particles.kind;
    var color = THEMES[this.themeName].colors["--accent"];
    ctx.clearRect(0, 0, width, height);
    for (var i = 0; i < this.particles.length; i += 1) {
      var particle = this.particles[i];
      this.step(particle, kind, width, height, now);
      this.paint(ctx, particle, kind, color, now);
    }
  },

  step: function (particle, kind, width, height, now) {
    var sway = Math.sin(now / 900 + particle.drift) * 0.35;
    if (kind === "petals" || kind === "embers") {
      particle.y -= particle.speed * (kind === "embers" ? 1.15 : 0.45);
      particle.x += sway;
      if (particle.y < -20) {
        particle.y = height + 10;
        particle.x = Math.random() * width;
      }
    } else if (kind === "rain") {
      particle.y += 2.4 + particle.speed * 3;
      particle.x += 0.35;
      if (particle.y > height + 10) {
        particle.y = -10;
        particle.x = Math.random() * width;
      }
    } else if (kind === "motes") {
      particle.y += Math.sin(now / 1400 + particle.drift) * 0.25;
      particle.x += Math.cos(now / 1600 + particle.drift) * 0.3;
      particle.spin += 0.008;
    } else if (kind === "sparks") {
      particle.x += Math.cos(particle.drift) * particle.speed * 0.8;
      particle.y += Math.sin(now / 500 + particle.drift) * 0.45;
      if (Math.sin(now / 700 + particle.spin) > 0.95) {
        particle.drift += 0.8;
      }
    } else {
      particle.x += Math.sin(now / 2000 + particle.drift) * 0.08;
      particle.y += 0.05;
    }
    if (particle.x < -20) particle.x = width + 10;
    if (particle.x > width + 20) particle.x = -10;
    if (particle.y > height + 30) particle.y = -10;
  },

  paint: function (ctx, particle, kind, color, now) {
    ctx.save();
    ctx.translate(particle.x, particle.y);
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.22 + particle.variant * 0.28;
    if (kind === "petals") {
      if (particle.variant > 0.72) {
        this.heart(ctx, particle.size);
      } else {
        ctx.rotate(particle.spin + now / 2000);
        ctx.beginPath();
        ctx.ellipse(0, 0, particle.size * 0.7, particle.size * 0.35, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (kind === "rain") {
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-1, particle.size * 2.4);
      ctx.stroke();
    } else if (kind === "embers") {
      ctx.globalAlpha = 0.35 + Math.sin(now / 180 + particle.drift) * 0.2;
      ctx.beginPath();
      ctx.arc(0, 0, particle.size * 0.45, 0, Math.PI * 2);
      ctx.fill();
    } else if (kind === "motes") {
      ctx.rotate(particle.spin);
      if (particle.variant > 0.8) {
        ctx.fillRect(-particle.size * 0.35, -particle.size * 0.2, particle.size * 0.7, particle.size * 0.4);
      } else {
        ctx.beginPath();
        ctx.arc(0, 0, particle.size * 0.4, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (kind === "sparks") {
      this.star(ctx, particle.size * 0.7);
    } else if (particle.variant > 0.86) {
      ctx.globalAlpha = 0.06;
      ctx.beginPath();
      ctx.arc(0, 0, particle.size * 8, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.globalAlpha = 0.25 + Math.sin(now / 600 + particle.drift) * 0.2;
      ctx.beginPath();
      ctx.arc(0, 0, particle.variant > 0.5 ? 1.4 : 0.8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  },

  heart: function (ctx, size) {
    var s = size * 0.55;
    ctx.beginPath();
    ctx.moveTo(0, s * 0.4);
    ctx.bezierCurveTo(-s, -s * 0.4, -s * 1.3, s * 0.6, 0, s * 1.3);
    ctx.bezierCurveTo(s * 1.3, s * 0.6, s, -s * 0.4, 0, s * 0.4);
    ctx.fill();
  },

  star: function (ctx, size) {
    ctx.beginPath();
    for (var i = 0; i < 8; i += 1) {
      var radius = i % 2 === 0 ? size : size * 0.4;
      var angle = (Math.PI / 4) * i - Math.PI / 2;
      var x = Math.cos(angle) * radius;
      var y = Math.sin(angle) * radius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
  },
};
