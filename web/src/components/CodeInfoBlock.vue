<script setup lang="ts">
import { onMounted } from "vue"
import Prism from "prismjs"
import "prismjs/components/prism-python"
import "prismjs/themes/prism.css"
import { ref } from 'vue';

onMounted(() => {
  Prism.highlightAll()
})

const manual_blocks = ref([
    {
        title: "Types",
        desc: `In this library, you will use pre-made custom types. These types include mat and vec, as shown in the example.

          • The mat type is a 2D Python list where each inner list represents one row of a matrix.
          • The vec type is a simple list of floating-point numbers.`,
        code: `
mat = list[list[float]]
vec = list[float]`
    },
    {
        title: "Errors",
        desc: `Your library should handle basic errors, which will be tested automatically.
          The custom errors defined for this library are:

          • ShapeMismatchedError — raised when the shape of a matrix does not match the expected shape
          • SingularError — raised when a matrix is singular to machine precision
        `,
        code: `
raise ShapeMismatchedError("Your custom message")
raise SingularError("Your custom message")
        `
    },
    {
      title: "Function Setup",
      desc: `For each task, you’ll get a short description, examples, and some theory.
          You’ll also be told what parameters your function receives and what it should return.
          Then, you’ll write your code inside the provided empty function.
          Do not modify the function signature.
      `,
      code: `
def lu(a: mat) -> tuple[mat, mat, mat]:
  # Do your awesome PA=LU decomposition

  return P, L, U
      `
    },
    {
      title: "Array references",
      desc: `Be careful not to modify the input parameters of your function — this will be tested.
          Most inputs are Python lists passed by reference, so changing them will also change the original matrix.
          If you need to modify a matrix, make a copy first using the provided copy() function.
      `,
      code: `
def lu(a: mat) -> tuple[mat, mat, mat]:
      a_copy = a          # Do not do this
      a_copy[0][0] = 0    # This will change a 

      a_copy = copy(a)    # Do this 
      a_copy[0][0] = 0     
      `
    },
    {
      title: "Reusability",
      desc: `In each function, you are allowed to reuse any functions you have already implemented.
However, no other libraries are available.
      `,
      code: `
def det(a: mat) -> float:
      P, L, U = lu(a)     # This is all right

      # Compute the determinant using P, L, U

      return det
      `
    }
]);

</script>

<template>
  <section class="px-6 md:px-20 lg:px-40 py-12 md:py-20 bg-gray-50">
    <div 
      v-for="block in manual_blocks" 
      :key="block.title"
      class="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-10 items-start mb-16"
    >
      <h3 class="col-span-1 md:col-span-2 text-2xl md:text-3xl font-bold">
        {{ block.title }}
      </h3>

      <p  class="text-gray-700 text-base md:text-lg leading-relaxed whitespace-pre-line">
        {{ block.desc }}
      </p>

      <div>
        <pre class="rounded-lg shadow p-4 overflow-x-auto bg-white">
          <code class="language-python">{{ block.code }}</code>
        </pre>
      </div>
    </div>
  </section>
</template>
