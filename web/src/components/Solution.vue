<script setup>
import Prism from "prismjs"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"
import { defineEmits, onMounted } from "vue"

onMounted(() => {
  Prism.highlightAll()
})

const emit = defineEmits(['close']);

defineProps({
    code: {
        type: String,
        required: true
    }
});
</script>

<template>
    <div class="fixed top-0 left-0 bottom-0 right-0 z-99 bg-black/20 flex" @click="emit('close')">
        <div class="bg-gray-50 p-10 m-auto rounded-3xl relative max-h-10/12" @click.stop>
            <p class="absolute top-5 right-5 w-full text-right text-4xl cursor-pointer" @click="emit('close')">ˣ</p>
            <pre class="rounded-2xl shadow m-4 p-4 bg-white whitespace-pre overflow-auto w-max max-h-140"><code class="language-python">{{ code }}</code></pre>
        </div>
    </div>
</template>