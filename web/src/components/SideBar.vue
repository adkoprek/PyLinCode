<script setup>
import { onBeforeMount, ref } from "vue";
import { RouterLink } from "vue-router";
import lessons_data from "../assets/lessons.json";
import { initDatabase, subscrbeToInsert, subscrbeToRemove, getLocks } from "@/scripts/db";

const isOpen = ref(false);
const locks = ref(new Set());
const lessons = lessons_data.lessons;

onBeforeMount(async () => {
  await initDatabase();
  const rawLocks = await getLocks();
  for (const lock of rawLocks) {
    locks.value.add(parseInt(lock.id));
  }

  subscrbeToInsert((id) => locks.value.add(parseInt(id)));
  subscrbeToRemove((id) => locks.value.delete(parseInt(id)))
});

defineProps({
  selected_id: {
    type: Number,
    required: true
  }
});
</script>

<template>
  <div :class="[
      'flex z-30 overflow-scroll wrapper',
      isOpen ? 'border-r-1 border-black' : ''
    ]">
    <div
      :class="[
        'text-gray-50 transition-all duration-400',
        isOpen ? 'w-50' : 'w-16',
      ]">
      <button
        class="p-4 w-full flex justify-center transition rounded-b-none text-gray-700 hover:bg-indigo-500 hover:text-white"
        @click="isOpen = !isOpen">
        ☰
      </button>
      <ul class="space-y-2">
        <div v-for="lesson in lessons" :key="lesson.id">
          <span v-if="locks.has(parseInt(lesson.id))" class="group relative">
            <li
              :class="[
                'flex items-center justify-center space-x-3 p-3 group opacity-50',
                isOpen ? 'justify-start pl-4' : 'justify-center']">
              <div
                class="flex items-center justify-center border w-8 h-8 rounded-full text-sm font-semibold text-gray-700 border-indigo-500">
                {{ lesson.id }}
              </div>
              <span v-if="isOpen" class="truncate font-medium text-gray-700">
                {{ lesson.title }}
              </span>
            </li>
          </span>
          <RouterLink v-else :to="`/coding/${parseInt(lesson.id)}`">
            <li
              :class="[
                'flex items-center justify-center space-x-3 cursor-pointer p-3 group',
                isOpen ? 'justify-start pl-4' : 'justify-center']">
              <div :class="[
                    'flex items-center justify-center border w-8 h-8 rounded-full text-sm font-semibold group-hover:bg-indigo-500 group-hover:text-white transition',
                    isOpen ? ' text-gray-700 border-indigo-500' : 'border-indigo-500 text-gray-700',
                    selected_id == lesson.id ? 'bg-indigo-500 text-white' : '']">
                {{ lesson.id }}
              </div>
              <span v-if="isOpen" class="truncate font-medium text-gray-700">
                {{ lesson.title }}
              </span>
            </li>
          </RouterLink>
        </div>
      </ul>
    </div>
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
