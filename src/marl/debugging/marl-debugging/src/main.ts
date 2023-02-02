import { createApp } from 'vue'
// import "@picocss/pico"
import "bootstrap/dist/css/bootstrap.min.css"
import App from './App.vue'
import { createPinia } from 'pinia'


const app = createApp(App);
app.use(createPinia());
app.mount('#app');
