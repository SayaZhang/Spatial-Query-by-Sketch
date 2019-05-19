<template xmlns:>
  <div id="app">
    <diagram />
    <globalMap />
    <candidateList/>
  </div>
</template>

<script>
  import GlobalMap from './components/globalMap'
  import Diagram from './components/diagram.vue'
  import CandidateList from './components/candidateList.vue'
  // import SearchGrid from './components/searchGrid'
  
  export default {
    components: {
      globalMap: GlobalMap,
      diagram: Diagram,
      candidateList: CandidateList
      // searchGrid: SearchGrid
    },
  }
</script>

<style lang="sass">
  @import ~bulma/sass/utilities/_all

  html, body, #app
    width: 100%
    height: 100%
    margin: 0
    padding: 0
</style>