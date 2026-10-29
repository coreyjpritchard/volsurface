// Prepended to marimo's Pyodide worker by scripts/build_public_vollab.py. It reports each
// download (the Python runtime, packages and data) to the loading screen in the page over
// a BroadcastChannel, counting bytes on a clone so the worker's own reads are untouched.
(() => {
  if (typeof BroadcastChannel === "undefined" || typeof self.fetch !== "function") return;
  const channel = new BroadcastChannel("vollab-loader");
  const post = (message) => {
    try {
      channel.postMessage(message);
    } catch (e) {}
  };
  const originalFetch = self.fetch.bind(self);
  self.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : (input && input.url) || String(input);
    post({ kind: "start", url });
    let response;
    try {
      response = await originalFetch(input, init);
    } catch (error) {
      post({ kind: "error", url, message: String(error) });
      throw error;
    }
    if (!response.ok) {
      post({ kind: "error", url, message: "HTTP " + response.status });
      return response;
    }
    try {
      const reader = response.clone().body.getReader();
      (async () => {
        let pending = 0;
        let lastPost = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          pending += value.byteLength;
          const now = Date.now();
          if (now - lastPost > 100) {
            post({ kind: "bytes", url, n: pending });
            pending = 0;
            lastPost = now;
          }
        }
        post({ kind: "done", url, n: pending });
      })().catch(() => {});
    } catch (e) {}
    return response;
  };
})();
