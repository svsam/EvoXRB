(() => {
  "use strict";

  const replay = document.querySelector("[data-evoxrb-replay]");
  if (!replay) {
    return;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== replay.contentWindow) {
      return;
    }
    if (!event.data || event.data.type !== "evoxrb-replay-height") {
      return;
    }
    const height = Number(event.data.height);
    if (Number.isFinite(height) && height >= 400 && height <= 2400) {
      replay.style.height = `${Math.ceil(height + 2)}px`;
    }
  });
})();
