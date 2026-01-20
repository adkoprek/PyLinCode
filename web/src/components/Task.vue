<script setup lang="ts">
import { ref, onMounted, watchEffect } from 'vue'

const props = defineProps({
  id: {
    type: Number,
    required: true
  }
});

const content = ref('')
const shadowHost = ref<HTMLDivElement | null>(null)
let shadow: ShadowRoot | null = null

onMounted(async () => {
  if (shadowHost.value) {
    shadow = shadowHost.value.attachShadow({ mode: 'open' })
    await loadContent()
  }
})

watchEffect(async () => {
  if (shadow) await loadContent()
})

async function loadContent() {
  if (!shadow) return
  const res = await fetch(`/desc/${props.id}_desc.html`)
  content.value = await res.text()

  shadow.innerHTML = `
    <link rel="stylesheet" href="https://stackedit.io/style.css">
    <style>h1 { margin: 0; }</style>
    ${content.value}`
}
</script>

<template>
  <div class="w-full flex justify-center pl-6 pr-2 py-6 h-full">
    <div class="w-full bg-white rounded-2xl shadow-xl p-6 pt-0 flex flex-col">
      <div class="overflow-y-scroll mt-9" ref="shadowHost"></div>
      <p class="bg-white pt-2">
        Powered by <a class="text-blue-400" href="https://stackedit.io">stackedit.io</a>
      </p>
    </div>
  </div>
</template>
