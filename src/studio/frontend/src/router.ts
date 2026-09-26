/**
 * Single route (`/`): sheets, drawers and overlays are state, not routes.
 */
import { createRouter, createWebHashHistory } from "vue-router";
import StudioPage from "./pages/StudioPage.vue";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", name: "studio", component: StudioPage },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});
