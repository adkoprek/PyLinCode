<script setup lang="ts">
import profile from '@/assets/profile.png';
</script>

<template>
  <section class="py-20 bg-gray-50">
    <div class="border-b border-gray-800 border-1 max-w-5xl mx-auto"></div>
    <div class="max-w-5xl mx-auto px-6 mt-12 flex flex-col md:flex-row items-center gap-12">
      
      <div class="flex-shrink-0">
        <img 
          :src="profile" 
          alt="Adam Koprek" 
          class="w-60 h-60 rounded-full object-cover shadow-lg"
        />
      </div>
      
      <div>
        <h2 class="text-3xl font-bold mb-4">About Me</h2>
        <h3 class="text-xl font-semibold text-gray-700 mb-2">Adam Koprek</h3>
        <p class="text-gray-600 leading-relaxed">

          I'm a high school student who enjoys programming and mathematics.
          In my first year of high school, I realized that most advanced programming relies heavily on linear algebra.
          So I took MIT’s 18.06 course taught by Gilbert Strang and started grinding.
          After mastering the theory, I decided to deepen my understanding by implementing a linear algebra library myself.
          I chose to make this library public on this website, because there is still so much to learn — and learn best by building.
        </p>
      </div>
    </div>
  </section>
</template>