<script setup>
import { ref, watchEffect } from 'vue';
import Task from '@/components/Task.vue';
import Editor from '@/components/Editor.vue';
import DragCol from "vue-resizer/DragCol.vue";
import Sidebar from '@/components/SideBar.vue';
import lessons_data from "../assets/lessons.json";

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
  ) || {
    id: -1,
    title: "Lesson Not Found",
    description: "The requested lesson does not exist.",
    code: ""
  };
});
</script>


<template>
    <div class="w-full flex bg-gray-50" style="height: calc(100vh - 3.5rem);">
        <Sidebar :selected_id="lesson.id" />
        <DragCol style="width: 100vw; height: calc(100vh - 3.5rem);" slider-bg-color="transparent" slider-bg-hover-color="transparent">
            <template #left>
                <Task :id="lesson.id" :title="lesson.title" />
            </template>
            <template #right>
                <Editor :id="lesson.id"/>
            </template>
        </DragCol>
    </div>
</template>