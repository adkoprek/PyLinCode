import { loadPyodide, FS } from "pyodide";


// Internal variables
let pyodide = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === "setup") {
    await initPyodide();
    self.postMessage({ type: "ready" });
  }

  if (type === "run") {
    await runCode(payload);
    self.postMessage({ type: "done" });
  }
}

async function initPyodide() {
  pyodide = await loadPyodide();
}

async function runCode(code) {
  try {
    const output = await pyodide.runPythonAsync(code.code);
    self.postMessage({ type: "out", payload: output });
  } catch (err) {
    const output = err + "\n";
    self.postMessage({ type: "err", payload: output });
  } 
}

async function test1(code) {
  FS.writeFile("/")
}
