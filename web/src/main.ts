import './assets/main.css'
import 'primeicons/primeicons.css'
import router from './router'

import { createApp } from 'vue'
import App from './App.vue'

import VueMathjax from 'vue-mathjax-next'

let app = createApp(App)
app.use(router)
app.use(VueMathjax)
app.mount('#app')
