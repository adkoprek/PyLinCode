import './assets/main.css'
import 'primeicons/primeicons.css'
import router from './router'

import { createApp } from 'vue'
import App from './App.vue'

import VueMathjax from 'vue-mathjax-next'

import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
(self as any).MonacoEnvironment = {
    getWorker(_: any, label: any) {
        return new editorWorker()
  }
};

let app = createApp(App)
app.use(router)
app.use(VueMathjax)
app.mount('#app')
