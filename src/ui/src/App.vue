<template>
  <main>
    <h1>RL dashboard</h1>
    <RouterView></RouterView>
    <footer>
      <SystemInfo />
    </footer>
  </main>
</template>


<script setup lang="ts">
import SystemInfo from './components/SystemInfo.vue';

</script>

<style>
:root {
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -webkit-text-size-adjust: 100%;
}


html,
body,
#app,
main {
  height: 100%;
  max-height: 100%;
}


main {
  overflow-x: hidden;
  padding-left: 1%;
  padding-right: 1%;
}

footer {
  position: fixed;
  bottom: 0;
  font-size: smaller;
  width: 100%;
  max-height: 2.5%;
}

dialog {
  top: 50%;
  left: 50%;
  translate: -50% -50%;
  border-radius: 3%;
}
</style>