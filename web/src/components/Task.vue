<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  id: Number,
});

const content = ref('')
const shadowHost = ref(null)

onMounted(async () => {
  const doc = await fetch(`/desc/${props.id}_desc.html`)
  content.value = await doc.text()

  const shadow = shadowHost.value.attachShadow({ mode: 'open' });
  shadow.innerHTML = `
      <link rel="stylesheet" href="https://stackedit.io/style.css">
      <style>h1 { margin: 0; }</style>` + content.value;
})
</script>

<template>
  <div class="w-full flex justify-center pl-6 pr-2 py-6 h-full">
    <div class="w-full bg-white rounded-2xl shadow-xl p-6 pt-0 flex flex-col">
      <div class="overflow-y-scroll mt-9" ref="shadowHost"></div>
      <p class="bg-white pt-2">Powered by <a class="text-blue-400" href="https://stackedit.io">stackedit.io</a></p>
    </div>
  </div>
</template>
