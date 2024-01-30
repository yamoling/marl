import { createApp } from 'vue'
import "bootstrap/dist/css/bootstrap.min.css"
import App from './App.vue'
import Experiment from './components/experiment/Experiment.vue'
import Home from './components/home/Home.vue'
import { createPinia } from 'pinia'
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

/* import all font awesome icons from the 'solid' and 'regular' families*/
import { fas } from '@fortawesome/free-solid-svg-icons'
import { far } from "@fortawesome/free-regular-svg-icons";
library.add(fas, far);

import { createRouter, createWebHashHistory } from "vue-router";



const router = createRouter(
    {
        history: createWebHashHistory(),
        routes: [
            {
                path: "/home",
                component: Home,
            },
            {
                path: "/",
                redirect: "/home",
            },
            {
                path: "/inspect/:logdir+",
                component: Experiment
            }
        ]
    }
)


createApp(App)
    .component("font-awesome-icon", FontAwesomeIcon)
    .use(createPinia())
    .use(router)
    .mount('#app');
