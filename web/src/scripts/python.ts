import lessons from "@/assets/lessons.json";
import { getCurrentCode, initDatabase } from "./db";
import { loadPyodide, PyodideAPI } from "pyodide"

let pyodide!: PyodideAPI;
const ctx = self as DedicatedWorkerGlobalScope;

type WorkerMessage =
  | { type: "setup"; payload: { id: number } }
  | { type: "run"; payload: { id: number; code: string } };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "done" }
  | { type: "out"; payload: string }
  | { type: "err"; payload: string };


const LESSON_MODULES: Record<number, string> = {
  1: "from src.vec_add import vec_add",
  2: "from src.vec_scl import vec_scl",
  3: "from src.vec_dot import vec_dot",
  4: "from src.vec_len import vec_len",
  5: "from src.vec_nor import vec_nor",
  6: "from src.mat_siz import mat_siz",
  7: "from src.mat_add import mat_add",
  8: "from src.mat_scl import mat_scl",
  9: "from src.mat_row import mat_row",
  10: "from src.mat_col import mat_col",
  11: "from src.mat_ide import mat_ide",
  12: "from src.mat_mul import mat_mul",
  13: "from src.mat_tra import mat_tra",
  14: "from src.mat_vec_mul import mat_vec_mul",
  15: "from src.lu import lu",
  16: "from src.solve import solve, for_sub, bck_sub",
  17: "from src.inv import inv",
  18: "from src.vec_prj import vec_prj",
  19: "from src.mat_prj import mat_prj",
  20: "from src.ortho import ortho",
  21: "from src.qr import qr",
  22: "from src.det import det",
}

function py() : PyodideAPI {
  if (!pyodide) throw new Error("First initialize pyodide");
  return pyodide
}

function post(msg: WorkerResponse) {
  ctx.postMessage(msg);
}

ctx.onmessage = async (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;

  switch (msg.type) {
    case "setup":
      await initPyodide(msg.payload.id);
      post({ type: "ready" });
      break;  

    case "run":
      await runCode(msg.payload.code, msg.payload.id);
      post({ type: "done" });
      break;
  }
}

async function initPyodide(lesson: number) {
  try{
    pyodide = await loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.1/full"
    });

    py().FS.mkdir("/src")
    py().FS.mkdir("/tests")

    await py().loadPackage("numpy");
    await loadPreviousLessons(lesson);
    await loadHelpers();
    await loadTest(lesson);
  } catch (err) {
    post({ type: "err", payload: String(err) });
  }
}

async function loadPreviousLessons(lesson: number) {
  await initDatabase();

  for (let i = 1; i < lesson; i++) {
    const lessonName = lessons.lessons[i - 1].title;
    const fileName = `/src/${lessonName}.py`;
    const code = await (await fetch(`/py/${i}_sol.py`)).text();
    py().FS.writeFile(fileName, addImports(code, i));
  }
}

async function fetchAndLoad(url: string, path: string) {
  const constsCode = await (await fetch(url)).text();
  py().FS.writeFile(path, constsCode);
}

async function loadHelpers() {
  await fetchAndLoad("/helpers/consts.py", "/tests/consts.py");
  await fetchAndLoad("/helpers/errors.py", "/src/errors.py");
  await fetchAndLoad("/helpers/types.py", "/src/types.py");
  await fetchAndLoad("/helpers/run_lesson.py", "/run_lesson.py");
  await fetchAndLoad("/helpers/hot_reload.py", "/hot_reload.py");
}

async function loadTest(lesson: number) {
  const testName = lessons.lessons[lesson - 1].title;
  const fileName = `/tests/lesson_${lesson}_test_${testName}.py`;
  await fetchAndLoad(fileName, fileName);
}

function addImports(code: string, lesson: number) {
  let final = ""

  let imports = "";
  for (let i = 1; i < lesson; i++) {
    imports += LESSON_MODULES[i] + "\n";
  }

  final += "from src.errors import ShapeMismatchedError, SingularError\n";
  final += "from src.types import vec, mat\n";
  final += "from copy import copy\n";
  final += imports;
  final += code;

  return final;
}

function loadCode(code: string, lesson: number) {
  const lessonName = lessons.lessons[lesson - 1].title;
  const fileName = `/src/${lessonName}.py`;
  const codeImport = addImports(code, lesson);
  py().FS.writeFile(fileName, codeImport);
}

async function runCode(code: string, lesson: number) {
  loadCode(code, lesson);

  try {
    await py().runPythonAsync(`
      import sys
      sys.path.append("/")
      import hot_reload
      import run_lesson
      hot_reload.hot_reload()
      run_lesson.run_lesson(${lesson})
    `);
    post({ type: "out", payload: `Lesson ${lesson} tests passed!` });
  } catch (err) {
    if (typeof err === "string")
      post({ type: "err", payload: err });

    else if (err instanceof Error) {
      post({ type: "err", payload: err.message });
   }
  }
}
