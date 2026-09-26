import { createPinia } from "pinia";
import { createApp } from "vue";
import App from "./App.vue";
import { resolveApi, setApi } from "./api";
import { router } from "./router";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/animations.css";
import "./styles/plots.css";

void resolveApi().then((api) => {
  setApi(api);
  document.documentElement.dataset.api = api.kind;
  createApp(App).use(createPinia()).use(router).mount("#app");
});
