<template>
  <div class="diagram">
    <div class="main">
      <tools class="menu"></tools>
      <div class="ground-panel">
        <operation></operation>
        <ground class="ground" ref="ground"></ground>
      </div>      
    </div>
  </div>
</template>

<script>
  import operation from "./operation.vue";
  import tools from "./tools.vue";
  import ground from "./ground.vue";
  export default {
    name: 'diagram',
    components:{
      operation,
      tools,
      ground
    },
    data () {
      return {

      }
    }
  }
</script>

<style lang="scss" scoped>
  .main{
    position: absolute;
    z-index: 9;
    display: flex;
    height: 100%;
  }
  .ground-panel {
    margin: 3em;
    .header{
      background: #4caf504f;
      margin-bottom: 10px;
    }
  }
</style>
