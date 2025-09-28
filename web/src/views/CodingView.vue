<script setup>
import Task from '@/components/Task.vue';
import Editor from '@/components/Editor.vue';
import DragCol from "vue-resizer/DragCol.vue";
import Sidebar from '@/components/SideBar.vue';
</script>


<template>
    <div class="w-full flex bg-gray-50">
        <Sidebar />
        <DragCol style="width: 100vw; height: calc(100vh - 3.5rem);" slider-bg-color="transparent" slider-bg-hover-color="transparent" left-percent="35">
            <template #left>
                <Task title="Test" description="This is a test task." />
            </template>
            <template #right>
                <Editor />
            </template>
        </DragCol>
    </div>
</template>