<template>
    <aside>
        <h1>Rl Debugger</h1>
        <nav>
            <ul class="list-group">
                    <li class="list-group-item" :class="item == selectedItem ? 'active' : ''" v-for="item in navItems"
                        @click="() => change(item)">
                        {{ item }}
                    </li>
            </ul>
        </nav>
    </aside>
</template>

<script setup lang="ts">
import { ref } from 'vue';
const navItems = [
    "File selection",
    "Training",
    "Replays"
]
const selectedItem = ref(navItems[0]);
const emit = defineEmits(["navChange"]);

function change(navItem: string) {
    selectedItem.value = navItem;
    emit("navChange", navItem);
}

defineExpose({ change });

</script>
