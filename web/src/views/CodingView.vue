<script setup>
import { ref, watchEffect } from 'vue';
import not_found from '@/assets/not_found.svg';
import Task from '@/components/Task.vue';
import Editor from '@/components/Editor.vue';
import DragCol from "vue-resizer/DragCol.vue";
import Sidebar from '@/components/SideBar.vue';
import lessons_data from "../assets/lessons.json";
import SubmissionsSideBar from '@/components/SubmissionsSideBar.vue';

const props = defineProps({
    id: {
        type: Number,
        required: true
    }
});

const lesson = ref({})

watchEffect(() => {
  lesson.value = lessons_data.lessons.find(
    lesson => parseInt(lesson.id) === props.id
  ) || { id: -1 };
});
</script>


<template>
    <div class="w-full flex bg-gray-50" style="height: calc(100vh - 3.5rem);">
        <Sidebar :selected_id="lesson.id" />
        <div v-if="lesson.id == -1" class="w-full flex justify-center pl-6 pr-2 py-6 h-full">
            <div class="w-full bg-white justify-center rounded-2xl shadow-xl p-6 pt-0 flex flex-col">
                <img :src="not_found" class="max-w-100 mx-auto" alt="Lesson Not Found" />
                <h2 class="text-center text-5xl mt-10">Lesson Not Found</h2>
            </div>
        </div>
        <DragCol v-else style="width: 100vw; height: calc(100vh - 3.5rem);" slider-bg-color="transparent" slider-bg-hover-color="transparent">
            <template #left>
                <Task :id="lesson.id" :title="lesson.title" />
            </template>
            <template #right>
                <Editor :id="lesson.id"/>
            </template>
        </DragCol>
    </div>
</template>