import { createApp } from 'vue'
import "bootstrap/dist/css/bootstrap.min.css"
import "bootstrap/dist/js/bootstrap.bundle.min.js"
import App from './App.vue'
import { createPinia } from 'pinia'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

/* import specific font awesome icons */
import { fas } from '@fortawesome/free-solid-svg-icons'
import { far } from "@fortawesome/free-regular-svg-icons";
library.add(fas, far);


const app = createApp(App);
app.component("font-awesome-icon", FontAwesomeIcon)
app.use(createPinia());
app.mount('#app');
