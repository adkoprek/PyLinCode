import lessons from "@/assets/lessons.json";
import { getCurrentCode, initDatabase } from "./db";
import { loadPyodide } from "pyodide"

let pyodide = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === "setup") {
    await initPyodide(payload.id);
    self.postMessage({ type: "ready" });
  }

  if (type === "run") {
    await runCode(payload.code, payload.id);
    self.postMessage({ type: "done" });
  }
}

async function initPyodide(lesson) {
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.1/full"
  });

  await pyodide.loadPackage("numpy");
  pyodide.FS.mkdir("/src")
  pyodide.FS.mkdir("/tests")
  await loadPreviousLessons(lesson);
  await loadHelpers();
  await loadTest(lesson);
}

async function loadPreviousLessons(lesson) {
  for (let i = 1; i < lesson; i++) {
    await initDatabase();
    const code = addImports((await getCurrentCode(i)).code);
    const lessonName = lessons.lessons[i - 1].title;
    const fileName = `/src/${lessonName}.py`;
    pyodide.FS.writeFile(fileName, code, { encoding: "utf8" });
  }
}

async function loadHelpers() {
  const constsCode = await (await fetch("/helpers/consts.py")).text();
  pyodide.FS.writeFile("/tests/consts.py", constsCode, { encoding: "utf8" });

  const errorsCode = await (await fetch("/helpers/errors.py")).text();
  pyodide.FS.writeFile("/src/errors.py", errorsCode, { encoding: "utf8" });

  const typesCode = await (await fetch("/helpers/types.py")).text();
  pyodide.FS.writeFile("/src/types.py", typesCode, { encoding: "utf8" });

  const runTestCode = await (await fetch("/helpers/run_lesson.py")).text();
  pyodide.FS.writeFile("/run_lesson.py", runTestCode, { encoding: "utf8" });

  const hotReloadCode = await (await fetch("/helpers/hot_reload.py")).text();
  pyodide.FS.writeFile("/hot_reload.py", hotReloadCode, { encoding: "utf8" });
}

async function loadTest(lesson) {
  const testName = lessons.lessons[lesson - 1].title;
  const fileName = `/tests/lesson_${lesson}_test_${testName}.py`;
  const code = await (await fetch(fileName)).text();
  pyodide.FS.writeFile(fileName, code, { encoding: "utf8" });
}

function addImports(code) {
  return `from src.errors import ShapeMismatchedError
from src.types import vec
${code}`;
}

async function runCode(code, lesson) {
  const lessonName = lessons.lessons[lesson - 1].title;
  const fileName = `/src/${lessonName}.py`;
  const codeImport = addImports(code);
  pyodide.FS.writeFile(fileName, codeImport, { encoding: "utf8" });

  try {
    await pyodide.runPythonAsync(`
      import sys
      sys.path.append("/")
      import hot_reload
      import run_lesson
      hot_reload.hot_reload()
      run_lesson.run_lesson(${lesson})
    `);
    self.postMessage({ type: "out", payload: `Lesson ${lesson} tests passed!` });
  } catch (err) {
    self.postMessage({ type: "err", payload: err.toString() });
  }
}
