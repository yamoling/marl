import { createApp } from 'vue'
import "bootstrap/dist/css/bootstrap.min.css"
import App from './App.vue'
import { createPinia } from 'pinia'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

/* import all font awesome icons from the 'solid' and 'regular' families*/
import { fas } from '@fortawesome/free-solid-svg-icons'
import { far } from "@fortawesome/free-regular-svg-icons";
library.add(fas, far);


const app = createApp(App);
app.component("font-awesome-icon", FontAwesomeIcon)
app.use(createPinia());
app.mount('#app');
