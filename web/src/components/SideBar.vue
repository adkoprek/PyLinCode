<script setup>
import { ref } from "vue";

const isOpen = ref(false);
const lessons = [
  { num: 1, title: "Introduction" },
  { num: 2, title: "Variables & Types" },
  { num: 3, title: "Control Flow" },
  { num: 4, title: "Functions" },
  { num: 5, title: "Data Structures" },
];

function toggleSidebar() {
  isOpen.value = !isOpen.value;
}

function handleClick(item) {
  console.log(item);
}
</script>

<template>
  <div class="flex bg-white z-30">
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
        <li
          v-for="lesson in lessons"
          :key="lesson.num"
          :class="[
            'flex items-center justify-center space-x-3 cursor-pointer p-3 group',
            isOpen ? 'justify-start pl-4' : 'justify-center'
          ]"
          @click="handleClick(lesson.title)"
        >
          <div
            :class="[
                'flex items-center justify-center border w-8 h-8 rounded-full text-sm font-semibold group-hover:bg-indigo-500 group-hover:text-white transition',
                isOpen ? ' text-white border-white' : 'border-indigo-500 text-gray-700'
            ]"
          >
            {{ lesson.num }}
          </div>

          <span v-if="isOpen" class="truncate font-medium">
            {{ lesson.title }}
          </span>
        </li>
      </ul>
    </div>
  </div>
</template>
