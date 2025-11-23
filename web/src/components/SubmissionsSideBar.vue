<script setup>
import { initDatabase, getSubmissions, subscribeToSubmissionInsert } from '@/scripts/db';
import { onMounted, ref, watch } from 'vue';
import Solution from './Solution.vue';
import {v4 as uuid} from "uuid"


const isOpen = ref(false);
const submissions = ref([]);
const overlay = ref(null);
const code = ref('');
const props = defineProps({
  id: {
    type: Number,
    required: true
  }
});

onMounted(async () => {
    await initDatabase();
    await subscribeToSubmissionInsert(submissionInsert)
    submissions.value = await getSubmissions(props.id);
});

watch(() => props.id, async (newId) => {
    submissions.value = await getSubmissions(newId);
});

async function submissionInsert(id) {
    if (props.id == parseInt(id)) {
        submissions.value = await getSubmissions(props.id);
    }
}

function openSubmission(submissionId) {
    const result = submissions.value.find((submission) => submission.id === submissionId);
    code.value = result.code;
    overlay.value = true;
}
</script>


<template>
    <Solution v-if="overlay" :code="code" @close="overlay = false" />
    <div
        :class="[
            'transition-all duration-400 overflow-y-auto wrapper',
            isOpen ? 'w-95 border-l-1 border-black' : 'w-16',
        ]">
        <button
            class="p-4 w-full flex justify-center transition-all rounded-b-none text-gray-700 hover:bg-indigo-500 hover:text-white"
            @click="isOpen = !isOpen">☰
        </button>
        <ul class="space-y-2">
        <p v-if="submissions.length == 0 && isOpen" class="w-full text-center mt-3 text-gray-700">No submissions yet</p>
        <div v-if="isOpen" v-for="submission in submissions" :key="submission.id">
            <li :class="[
                'flex items-center justify-center space-x-3 p-3 group cursor-pointer hover:bg-indigo-500 group transition',
                isOpen ? 'justify-start pl-4' : 'justify-center']"
                @click="openSubmission(submission.id)">
                <span v-if="isOpen" class="truncate font-medium text-gray-700 group-hover:text-white">
                    Submission on
                    {{ new Date(submission.timestamp).toLocaleString() }}
                </span>
            </li>
        </div>
        </ul>
    </div>
</template>

<style>
.wrapper::-webkit-scrollbar {
  display: none;
}

.wrapper {
  -ms-overflow-style: none; 
  scrollbar-width: none;
}
</style>