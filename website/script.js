const copyButtons = document.querySelectorAll("[data-copy]");

async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const fallback = document.createElement("textarea");
  fallback.value = value;
  fallback.setAttribute("readonly", "");
  fallback.style.position = "fixed";
  fallback.style.opacity = "0";
  document.body.append(fallback);
  fallback.select();

  const copied = document.execCommand("copy");
  fallback.remove();

  if (!copied) {
    throw new Error("Copy command was not available");
  }
}

copyButtons.forEach((button) => {
  const originalLabel = button.textContent;
  let resetTimer;

  button.hidden = false;

  button.addEventListener("click", async () => {
    window.clearTimeout(resetTimer);
    const status = button.getAttribute("aria-describedby");
    const statusElement = status ? document.getElementById(status) : null;

    try {
      await copyText(button.dataset.copy);
      button.textContent = "Copied";
      if (statusElement) statusElement.textContent = "Installation command copied.";
    } catch {
      button.textContent = "Copy failed";
      if (statusElement) {
        statusElement.textContent = "Copy failed. Select the installation command manually.";
      }
    }

    resetTimer = window.setTimeout(() => {
      button.textContent = originalLabel;
    }, 1600);
  });
});
