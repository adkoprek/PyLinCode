<script setup lang="ts">
import not_found from '@/assets/not_found.svg';
</script>


<template>
    <div style="height: calc(100vh - 3.5rem);" class="flex flex-col justify-center align-middle">
        <img :src="not_found" class="max-w-100 mx-auto" alt="Lesson Not Found" />
        <h2 class="text-center text-5xl mt-10">Page Not Found</h2>
    </div>
</template>