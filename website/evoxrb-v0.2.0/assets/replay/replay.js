(() => {
  "use strict";

  const manifest = window.EVOXRB_REPLAY;
  if (!manifest || !Array.isArray(manifest.frames) || manifest.frames.length === 0) {
    throw new Error("EvoXRB replay manifest is missing or empty");
  }

  const frame = document.querySelector("#replay-frame");
  const slider = document.querySelector("#generation-slider");
  const output = document.querySelector("#generation-output");
  const status = document.querySelector("#replay-status");
  const announcement = document.querySelector("#replay-announcement");
  const speed = document.querySelector("#replay-speed");
  const loop = document.querySelector("#replay-loop");
  const buttons = new Map(
    [...document.querySelectorAll("[data-action]")].map((button) => [
      button.dataset.action,
      button,
    ]),
  );

  let current = 0;
  let timer = null;

  const maximum = manifest.frames.length - 1;
  slider.max = String(maximum);
  slider.setAttribute("aria-valuemax", String(maximum));

  function announceHeight() {
    window.parent.postMessage(
      { type: "evoxrb-replay-height", height: document.documentElement.scrollHeight },
      "*",
    );
  }

  function render(index, shouldAnnounce = true) {
    current = Math.max(0, Math.min(maximum, Number(index)));
    frame.src = manifest.frames[current];
    frame.alt = `${manifest.alt_prefix} ${current}`;
    slider.value = String(current);
    slider.setAttribute("aria-valuenow", String(current));
    slider.setAttribute("aria-valuetext", `Generation ${current} of ${maximum}`);
    output.value = `${current} / ${maximum}`;
    status.textContent = `Generation ${current} of ${maximum}`;
    if (shouldAnnounce) {
      announcement.textContent = `Generation ${current} of ${maximum}`;
    }
    announceHeight();
  }

  function pause() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  function next(shouldAnnounce = true) {
    if (current < maximum) {
      render(current + 1, shouldAnnounce);
      return;
    }
    if (loop.checked) {
      render(0, shouldAnnounce);
    } else {
      pause();
      announcement.textContent = `Playback finished at generation ${maximum}`;
    }
  }

  function play() {
    pause();
    const framesPerSecond = Math.max(1, Number(speed.value));
    announcement.textContent = `Playback started at generation ${current}`;
    timer = window.setInterval(() => next(false), 1000 / framesPerSecond);
  }

  slider.addEventListener("input", () => {
    pause();
    render(slider.value);
  });
  speed.addEventListener("change", () => {
    if (timer !== null) {
      play();
    }
  });

  buttons.get("first").addEventListener("click", () => {
    pause();
    render(0);
  });
  buttons.get("previous").addEventListener("click", () => {
    pause();
    render(current - 1);
  });
  buttons.get("play").addEventListener("click", play);
  buttons.get("pause").addEventListener("click", () => {
    pause();
    announcement.textContent = `Playback paused at generation ${current}`;
  });
  buttons.get("next").addEventListener("click", () => {
    pause();
    next();
  });
  buttons.get("last").addEventListener("click", () => {
    pause();
    render(maximum);
  });

  frame.addEventListener("load", announceHeight);
  window.addEventListener("resize", announceHeight);
  window.addEventListener("pagehide", pause);
  render(0);
})();
