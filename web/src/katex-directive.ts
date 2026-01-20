import katex from "katex";
import "katex/dist/katex.min.css";

export default {
  mounted(el: any, binding: any) {
    katex.render(binding.value, el, {
      throwOnError: false,
      displayMode: false,
    });
  },
  updated(el: any, binding: any) {
    katex.render(binding.value, el, {
      throwOnError: false,
      displayMode: false,
    });
  },
};
