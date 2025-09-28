<script setup>
import { ref } from "vue";
import { RouterLink } from "vue-router";
import lessons_data from "../assets/lessons.json";

const isOpen = ref(false);
const lessons = lessons_data.lessons;

defineProps({
  selected_id: {
    type: Number,
    required: true
  }
});

function toggleSidebar() {
  isOpen.value = !isOpen.value;
}

</script>

<template>
  <div :class="[
    'flex z-30 overflow-scroll',
    isOpen ? 'bg-indigo-600' : 'bg-transparent',
  ]">
    <div
      :class="[
        'text-gray-50 transition-all duration-400',
        isOpen ? 'w-64' : 'w-16',
        isOpen ? 'bg-indigo-600' : 'bg-transparent',
      ]">
      <button
        :class="[
            'p-4 w-full flex justify-center transition rounded-b-none',
           isOpen ? 'text-white hover:bg-indigo-500' : 'text-gray-700 hover:text-indigo-600'
        ]"
        @click="toggleSidebar">
        ☰
      </button>
      <ul class="space-y-2">
        <RouterLink :to="`/coding/${parseInt(lesson.id)}`" v-for="lesson in lessons" :key="lesson.id">
          <li
            :class="[
              'flex items-center justify-center space-x-3 cursor-pointer p-3 group',
              isOpen ? 'justify-start pl-4' : 'justify-center'
            ]"
          >
            <div
              :class="[
                  'flex items-center justify-center border w-8 h-8 rounded-full text-sm font-semibold group-hover:bg-indigo-500 group-hover:text-white transition',
                  isOpen ? ' text-white border-white' : 'border-indigo-500 text-gray-700',
                  selected_id == lesson.id ? 'bg-indigo-500 text-white' : ''
              ]"
            >
              {{ lesson.id }}
            </div>

            <span v-if="isOpen" class="truncate font-medium">
              {{ lesson.title }}
            </span>
          </li>
        </RouterLink>
      </ul>
    </div>
  </div>
</template>
