<script setup>
import { onMounted, ref, watchEffect } from 'vue';
import not_found from '@/assets/not_found.svg';
import Task from '@/components/Task.vue';
import Editor from '@/components/Editor.vue';
import DragCol from "vue-resizer/DragCol.vue";
import Sidebar from '@/components/SideBar.vue';
import lessons_data from "../assets/lessons.json";
import SubmissionsSideBar from '@/components/SubmissionsSideBar.vue';
import { getLocks, initDatabase } from '@/scripts/db';

const props = defineProps({
    id: {
        type: Number,
        required: true
    }
});

const lesson = ref({})
const locked = ref(false);

onMounted(async () => {
    await initDatabase();
    const locks = await getLocks();
    locks.forEach((e) => {
        if (props.id == e.id) locked.value = true;
    });
});

watchEffect(() => {
    locked.value = false;
    lesson.value = lessons_data.lessons.find(
        lesson => parseInt(lesson.id) === props.id
    ) || { id: -1 };
});
</script>


<template>
    <div class="w-full flex bg-gray-50" style="height: calc(100vh - 3.5rem);">
        <Sidebar :selected_id="lesson.id" />

        <div class="relative flex-1 h-full">

            <!-- LOCK OVERLAY (only over content, not sidebar) -->
            <div
                v-if="locked"
                class="absolute inset-0 z-40 flex items-center justify-center bg-gray-900/60 backdrop-blur-sm"
            >
                <div class="bg-white rounded-2xl shadow-2xl px-10 py-8 text-center max-w-md">
                    <h2 class="text-3xl font-bold mb-3">Lesson Locked</h2>
                    <p class="text-gray-600 mb-6">
                        Complete the previous lesson to unlock this one.
                    </p>
                </div>
            </div>

            <!-- Lesson not found -->
            <div
                v-if="lesson.id == -1"
                class="w-full flex justify-center pl-6 pr-2 py-6 h-full"
            >
                <div class="w-full bg-white justify-center rounded-2xl shadow-xl p-6 pt-0 flex flex-col">
                    <img :src="not_found" class="max-w-100 mx-auto" alt="Lesson Not Found" />
                    <h2 class="text-center text-5xl mt-10">Lesson Not Found</h2>
                </div>
            </div>

            <DragCol
                v-else
                class="h-full"
                style="width: 100%; height: 100%;"
                slider-bg-color="transparent"
                slider-bg-hover-color="transparent">
                <template #left>
                    <Task :id="lesson.id" :title="lesson.title" />
                </template>
                <template #right>
                    <Editor :id="lesson.id" :locked="locked" />
                </template>
            </DragCol>
        </div>

        <SubmissionsSideBar :id="props.id"></SubmissionsSideBar>
    </div>
</template>