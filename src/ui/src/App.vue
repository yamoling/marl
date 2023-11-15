<template>
  <main>
    <header class="row mb-1">
      <h1 class="col"> Experiment manager</h1>
    </header>
    <RouterView></RouterView>
    <!-- <Home /> -->
    <footer class="row">
      <SystemInfo />
    </footer>
  </main>
</template>


<script setup lang="ts">
import Home from './components/home/Home.vue';
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
body {
  height: 100%;
  margin: 0;
}

#app {
  height: 100%;
  width: 100%;
  padding: 1%;
  display: flex;
  flex-direction: row;
}

main {
  width: 100%;
  height: 100%;
  padding-left: 0.5%;
  /* overflow: hidden; */
}

footer {
  position: fixed;
  bottom: 0;
  font-size: smaller;
  width: 100%;
}

dialog {
  top: 50%;
  left: 50%;
  translate: -50% -50%;
  border-radius: 3%;
}
</style>