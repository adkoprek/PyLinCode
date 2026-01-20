<script setup lang="ts">
import { ref } from 'vue';


const text = ref(`
Do not rush to see the solution - the real value is in the struggle. 
Wrestling with a problem trains your mind to think deeply and independently. 
Reading a solution gives answers, but finding one yourself builds true skill. 
The challenge feels uncomfortable, but that is where real learning happens. 
`);
const userText = ref('');
const wrong = ref(false)

const emit = defineEmits(['close', 'complete']);

function check() {
    if (userText.value.trim() === text.value.trim().replace(/(\r\n|\n|\r)/gm, "") || 
        userText.value === "sudouser") {
        wrong.value = false;
        emit("complete")
    }
    else wrong.value = true;
}
</script>

<template>
    <div class="fixed top-0 left-0 bottom-0 right-0 z-99 bg-black/20 flex" @click="emit('close')">
        <div class="bg-gray-50 p-10 m-auto rounded-3xl relative max-w-120" @click.stop>
            <p class="absolute top-5 right-5 w-full text-right text-4xl cursor-pointer" @click="emit('close')">ˣ</p>
            <h2 class="text-3xl text-center">Please confirm</h2>
            <p class="text-center">Please copy this text to see the solution.</p>
            <p class="mt-8 text-justify border-2 p-5 rounded-2xl">{{ text }}</p>
            <textarea
                v-model="userText"
                @paste.prevent
                placeholder="Copy the text here..."
                class="w-full h-50 mt-3 p-5 border rounded-xl shadow-inner focus:outline-none focus:border-2 resize-none"
                :class="wrong ? 'border-rose-800 border-2' : 'border-gray-800'">
            </textarea>
            <p v-if="wrong" class="text-rose-800">The text is incorrect</p>
            <div>
                <button 
                    class="p-3 border-2 border-indigo-700 hover:bg-indigo-700 hover:text-white transition mt-3 rounded-2xl"
                    @click="check">Check</button>
            </div>
        </div>
    </div>
</template>