/**
 * Home selects a server workspace; the studio route requires an explicit entry this session.
 */
import { createRouter, createWebHashHistory } from "vue-router";
import StudioPage from "./pages/StudioPage.vue";
import HomePage from "./pages/HomePage.vue";
import { useNamedWorkspacesStore } from "./stores/namedWorkspaces";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", name: "home", component: HomePage },
    { path: "/studio", name: "studio", component: StudioPage },
    { path: "/:pathMatch(.*)*", redirect: "/" },
  ],
});

router.beforeEach((to) => {
  if (to.name === "studio" && !useNamedWorkspacesStore().activeId) return { name: "home" };
});
