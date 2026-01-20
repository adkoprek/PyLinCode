import { createRouter, createWebHistory } from 'vue-router';
import HomeView from '@/views/HomeView.vue';
import ManualView from '@/views/ManualView.vue';
import CodingView from '@/views/CodingView.vue';
import NotFoundView from '@/views/NotFoundView.vue';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/manual',
      name: 'manual',
      component: ManualView 
    },
    {
      path: '/coding/:id',
      name: 'coding',
      component: CodingView,
      props: route => ({ id: Number(route.params.id) })
    },
    {
      path: '/:catchAll(.*)',
      name: 'not-found',
      component: NotFoundView
    }
  ]
});

export default router;
