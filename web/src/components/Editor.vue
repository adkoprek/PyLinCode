<script setup lang="ts">
import { CodeEditor } from "monaco-editor-vue3";
import { VueSpinnerIos } from "vue3-spinners";
import DragRow from "vue-resizer/DragRow.vue";
import * as monaco from "monaco-editor";
import { ref, onMounted, watch, Ref, onUnmounted } from "vue";
import lessons_data from "@/assets/lessons.json"
import SolutionChallange from "./SolutionChallange.vue";
import Solution from "./Solution.vue";
import { changeCurrent, getCurrentCode, initDatabase, addSubmition } from "@/scripts/db";
import debounce from "lodash.debounce";
import { v4 as uuidv4 } from "uuid";


const editorOptions = {
  automaticLayout: true,
  fontSize: 12,
  minimap: { enabled: false },
};

const code = ref("");
const output: Ref<{ type: string; text: string }[]> = ref([]);

const isRunning = ref(false);
const isLoading = ref(false);

const showChallenge = ref(false);
const showSolution = ref(false);

let solution: string = "";
let challanged: boolean = false;

let loaded = false;
let worker!: Worker;

const props = defineProps({
  id: {
    type: Number,
    required: true,
  },
});

onMounted(async () => {
  await initDatabase();
  await loadCode(props.id);
  initMonaco();
  initPyodite();
});


watch(props, async () => {
  await loadCode(props.id);
  output.value = [];
  loaded = false;
})

const debounceFunc = debounce(async function () {
  await changeCurrent({ id: props.id.toString(), code: code.value });
}, 500);

watch(code, () => {
  debounceFunc();
});

const interval = setInterval(async () => {
  await changeCurrent({ id: props.id.toString(), code: code.value });
}, 10000);

onUnmounted(() => {
  clearInterval(interval);
});


/****************************** EDITOR ********************************/

async function loadCode(id: number) {
  let lesson = lessons_data.lessons.find(l => l.id == id);
  if (!lesson) return;
  
  const res = await fetch(`/py/${lesson.id}_sol.py`);
  solution = await res.text();

  const current = await getCurrentCode(id);
  if (current == null) {
    code.value = solution.split("\n")[0];
  }
  else {
    code.value = current.code;
  }
}

function initMonaco() {
  monaco.editor.defineTheme("my", {
    base: 'vs',
    inherit: true,
    rules: [],
    colors: {
      'editor.background': '#ffffff'
    }
  });
}

function scrollToBottom() {
  const terminal = document.getElementById("terminal");
  if (!terminal) return;
  terminal.scrollTop = terminal.scrollHeight;
}

/****************************** EXEC ********************************/

function initPyodite() {
  worker = new Worker(new URL('@/scripts/python.ts', import.meta.url), { type: 'module' });

  worker.onmessage = async (e) => {
    const { type, payload } = e.data;
    if (type === 'ready') {
      isLoading.value = false;
      loaded = true;
      runCode();
    }
    else if (type === 'out') await nextLesson(payload);
    else if (type === 'err') output.value.push({ type: 'text-red-500', text: payload });
    else if (type === 'done') isRunning.value = false;

    scrollToBottom();
  }
}

function runCode() {
  if (!loaded) {
    isLoading.value = true;
    worker.postMessage({ type: 'setup', payload: { id: props.id } });
    return;
  }

  if (isRunning.value) return;
  output.value = [];
  isRunning.value = true;
  worker.postMessage({ type: "run", payload: { code: code.value, id: props.id } });
}


async function nextLesson(payload: string) {
  await addSubmition({
    id: uuidv4(),
    lessonId: props.id,
    code: code.value,
    timestamp: new Date().toISOString()
  })

  output.value.push({ type: 'text-black', text: payload });
}

/****************************** SOLUTION ********************************/

function toggleSolution() {
  if (!challanged) {
    showChallenge.value = true;
    return;
  }
  showSolution.value = true;
}

function challangeComplete() {
  showChallenge.value = false;
  challanged = true;
  toggleSolution();
}
</script>

<template>
  <div class="w-full h-full flex pr-6 pl-2 py-6">
    <SolutionChallange 
      v-if="showChallenge"
      @complete="challangeComplete"
      @close="showChallenge = false"/>
    <Solution 
      v-if="showSolution" 
      :code="solution" 
      @close="showSolution = false"/> 
    <div
      class="w-full bg-white rounded-2xl shadow-xl p-6 flex flex-col">
      <div class="flex flex-col flex-1 h-screen">
        <DragRow class="flex-1" slider-bg-color="transparent" slider-bg-hover-color="trasparent" style="width: 100%;">
          <template #top>
            <div class="relative h-full overflow-hidden rounded-xl border shadow-inner">
              <CodeEditor
                class="py-3 h-full"
                v-model:value="code"
                language="python"
                theme="my"
                :options="editorOptions"
              />
            </div>
          </template>
          <template #bottom>
            <div v-if="isLoading"
              class="flex flex-col items-center justify-center align-middle flex-1 space-y-4 rounded-xl border shadow-inner h-full">
              <VueSpinnerIos size="50" color="#4F46E5" />
              <div class="text-gray-600 text-lg font-medium">
                Loading Python environment...
              </div>
            </div>

            <div v-else class="w-full h-full relative">
              <div id="terminal" class="absolute top-0 bottom-0 left-0 right-0 font-mono p-4 rounded-xl shadow-inner border overflow-auto">
                <pre v-for="(line, index) in output" :key="index" :class="line.type">{{ line.text }}</pre>
              </div>
            </div>
          </template>
        </DragRow>
        
        <div class="flex justify-between mt-3">
          <button
            @click="toggleSolution"
            class="px-6 py-2 bg-transparent rounded-xl shadow hover:bg-indigo-700 hover:text-white border-2 border-indigo-700 disabled:opacity-50 transition">
            Show Solution
          </button>
          <button
            @click="runCode"
            class="px-6 py-2 bg-indigo-600 text-white rounded-xl shadow hover:bg-indigo-700 disabled:opacity-50 transition"
            :disabled="isRunning">
            {{ isRunning ? "Running..." : "Run Code" }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>