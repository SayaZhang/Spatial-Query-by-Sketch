<template>
  <div class="header">
    <div class="operation">
      <template v-if="! isSelected">
        <div class="clear subOperation" @mouseenter="enter" @mouseleave="leave">
          <div class="theIcon" @click="clearCanvas">
            <svg class="icon icon-M fill-undefined undefined">
              <use xlink:href="#icon-delete-all">
                <svg id="icon-delete-all" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>delete-all</title>
                  <path :fill="svgColor" d="M8 25.333c0 1.467 1.2 2.667 2.667 2.667h10.667c1.467 0 2.667-1.2 2.667-2.667v-16h-16v16zM25.333 5.333h-4.667l-1.333-1.333h-6.667l-1.333 1.333h-4.667v2.667h18.667v-2.667z"></path>
                </svg>
              </use>
            </svg>
          </div>
          <div class="text" @click="clearCanvas">Clear all</div>
        </div>
        <!-- <div class="toggle" @click="changeGird">
          <template v-if="! hasGrid">
            <div class="theIcon">
              <svg class="icon icon-M fill-undefined undefined">
                <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-grid-off">
                  <svg id="icon-grid-off" viewBox="0 0 32 32" width="100%" height="100%">
                    <title>grid-off</title>
                    <path d="M10.667 5.333v1.933l2.667 2.667v-4.6h5.333v5.333h-4.6l2.667 2.667h1.933v1.933l2.667 2.667v-4.6h5.333v5.333h-4.6l2.667 2.667h1.933v1.933l2.667 2.667v-20.6c0-1.467-1.2-2.667-2.667-2.667h-20.6l2.667 2.667h1.933zM21.333 5.333h5.333v5.333h-5.333v-5.333zM1.693 1.693l-1.693 1.707 2.667 2.667v20.6c0 1.467 1.2 2.667 2.667 2.667h20.613l2.667 2.667 1.693-1.693-28.613-28.613zM13.333 16.733l1.933 1.933h-1.933v-1.933zM5.333 8.733l1.933 1.933h-1.933v-1.933zM10.667 26.667h-5.333v-5.333h5.333v5.333zM10.667 18.667h-5.333v-5.333h4.6l0.733 0.733v4.6zM18.667 26.667h-5.333v-5.333h4.6l0.733 0.72v4.613zM21.333 26.667v-1.947l1.947 1.947h-1.947z"></path>
                  </svg>
                </use>
              </svg>
            </div>
          </template>
          <template v-else>
            <div class="theIcon">
              <svg class="icon icon-M fill-undefined undefined">
                <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-grid-on">
                  <svg id="icon-grid-on" viewBox="0 0 32 32" width="100%" height="100%">
                    <title>grid-on</title>
                    <path d="M26.667 2.667h-21.333c-1.467 0-2.667 1.2-2.667 2.667v21.333c0 1.467 1.2 2.667 2.667 2.667h21.333c1.467 0 2.667-1.2 2.667-2.667v-21.333c0-1.467-1.2-2.667-2.667-2.667zM10.667 26.667h-5.333v-5.333h5.333v5.333zM10.667 18.667h-5.333v-5.333h5.333v5.333zM10.667 10.667h-5.333v-5.333h5.333v5.333zM18.667 26.667h-5.333v-5.333h5.333v5.333zM18.667 18.667h-5.333v-5.333h5.333v5.333zM18.667 10.667h-5.333v-5.333h5.333v5.333zM26.667 26.667h-5.333v-5.333h5.333v5.333zM26.667 18.667h-5.333v-5.333h5.333v5.333zM26.667 10.667h-5.333v-5.333h5.333v5.333z"></path>
                  </svg>
                </use>
              </svg>
            </div>
          </template>
          <div class="text">Toggle grid</div>
        </div> -->
      </template>
      <template v-else>
        <div class="clear subOperation" @mouseenter="enter" @mouseleave="leave">
          <div class="theIcon" @click="deleteObj">
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-remove-circle-outline">
                <svg id="icon-remove-circle-outline" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>remove-circle-outline</title>
                  <path :fill="svgColor" d="M9.333 14.667v2.667h13.333v-2.667h-13.333zM16 2.667c-7.36 0-13.333 5.973-13.333 13.333s5.973 13.333 13.333 13.333 13.333-5.973 13.333-13.333-5.973-13.333-13.333-13.333zM16 26.667c-5.88 0-10.667-4.787-10.667-10.667s4.787-10.667 10.667-10.667 10.667 4.787 10.667 10.667-4.787 10.667-10.667 10.667z"></path>
                </svg>
              </use>
            </svg>
          </div>
          <div class="text" @click="deleteObj">Delete</div>
        </div>
        <div class="rotate subOperation" v-show="! isLine">
          <div class="theIcon" @click="rotateObj">
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-rotate">
                <svg id="icon-rotate" viewBox="0 0 32 32" width="100%" height="100%">
                  <title>rotate</title>
                  <path d="M9.787 8.547l-8.64 8.653 8.653 8.64 8.653-8.64-8.667-8.653zM4.92 17.2l4.88-4.88 4.867 4.88-4.88 4.88-4.867-4.88zM25.813 8.853c-2.333-2.347-5.413-3.52-8.48-3.52v-4.32l-5.653 5.653 5.653 5.653v-4.32c2.387 0 4.773 0.907 6.6 2.733 3.64 3.64 3.64 9.56 0 13.2-1.827 1.827-4.213 2.733-6.6 2.733-1.293 0-2.587-0.28-3.787-0.813l-1.987 1.987c1.8 0.987 3.787 1.493 5.773 1.493 3.067 0 6.147-1.173 8.48-3.52 4.693-4.68 4.693-12.28 0-16.96z"></path>
                </svg>
              </use>
            </svg>
          </div>
          <div class="text" @click="rotateObj">Rotate</div>
        </div>
        <div class="subOperation" :class="toScale ? 'scale' : ''" v-show="! isLine">
          <div class="theIcon" @click="scaleObj">
            <svg class="icon icon-M fill-undefined undefined">
              <use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="#icon-scale">
                <svg id="icon-scale" viewBox="0 0 1024 1024" width="90%" height="90%">
                  <title>scale</title>
                  <path d="M957.293714 0.146286H66.56a66.194286 66.194286 0 0 0-66.486857 65.828571v891.830857c0 36.352 29.769143 65.828571 66.486857 65.828572h890.660571a66.194286 66.194286 0 0 0 66.413715-65.828572V66.048c0-36.425143-29.696-65.901714-66.413715-65.901714z m4.973715 921.161143c0 31.963429-9.142857 40.886857-41.398858 40.886857H103.058286c-32.329143 0-41.398857-12.288-41.398857-44.324572V102.546286c0-32.036571 9.069714-40.96 41.398857-40.96h817.737143c32.329143 0 41.472 8.923429 41.472 40.96v818.761143z"  /><path d="M857.234286 131.072H560.274286c-19.456 0-35.108571 13.458286-35.108572 32.914286 0 0 2.779429 27.648 32.768 27.648h222.646857L549.888 423.058286l51.346286 51.565714 229.668571-230.4v215.990857c0 30.208 29.915429 32.475429 29.915429 32.475429 19.382857 0 31.670857-15.798857 31.670857-35.254857v-291.108572a35.254857 35.254857 0 0 0-35.181714-35.254857zM163.986286 896.073143h297.106285c19.456 0 35.181714-13.458286 35.181715-32.914286 0 0-2.779429-27.648-32.768-27.648H240.786286L471.478857 604.16l-51.419428-51.565714-229.668572 230.4V567.003429c0-30.208-29.915429-32.475429-29.915428-32.475429-19.382857 0-31.597714 15.798857-31.597715 35.254857v291.108572c0 19.456 15.725714 35.254857 35.108572 35.254857z"  />
                  </svg>
              </use>
            </svg>
          </div>
          <div class="text" @click="scaleObj">Scale</div>
        </div>
        <template v-if="playerText">
          <label for="playerText"></label>
          <input id="playerText" class="playerText" v-model="text" ref="input" :placeholder="isPlayer ? '' : 'Label'" :maxlength="isPlayer ? 10 : 2" />
        </template>
      </template>
    </div>
    <div class="save">
      <span @click="download">Search</span>
    </div>
  </div>
</template>

<script>
  export default {
    name: 'operation',
    data () {
      return {
        hasBorder:false,
        svgColor:'black'
      }
    },
    computed:{
      clearState(){
        return this.$store.state.clearState;
      },
      hasGrid(){
        return this.$store.state.hasGrid;
      },
      isSelected(){
        return this.$store.state.isSelected;
      },
      playerText(){
        return this.$store.state.playerText;
      },
      isLine(){
        return this.$store.state.isLine;
      },
      isPlayer(){
        return this.$store.state.isPlayer;
      },
      toScale(){
        return this.$store.state.toScale;
      },
      text : {
        get(){
          return this.$store.state.text;
        },
        set(value){
          this.$store.commit('setText',value);
        }
      }
    },
    methods:{
      download(){
        this.$store.commit('changeDownloading');
      },
      clearCanvas(){
        this.$store.commit('changeToClear');
      },
      deleteObj(){
        this.$store.commit('changeToDelete');
      },
      changeGird(){
        this.$store.commit('changeGirdState');
      },
      rotateObj(){
        this.$store.commit('changeToRotate');
      },
      scaleObj(){
        this.$store.commit('changeToScale');
      },
      goBack(){
        this.$router.goBack();
      },
      enter(){
        this.svgColor = '#d1495b'
      },
      leave(){
        this.svgColor = 'black';
      }
    }
  }
</script>

<style lang="scss" scoped>
  .icon{
    width: 24px;
    height: 24px;
  }
  .header{
    width: 100%;
    height: 46px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    .back{
      width: 54px;
      height: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor:pointer
    }
    .operation{
      display: flex;
      margin-left: 5px;
      .theIcon{
        margin-top: 3px;
      }
      .text{
        line-height: normal;
        font-size: 12px;
      }
      .subOperation{
        cursor:pointer;
        display: flex;
        align-items: center;
        text-align: center;
        background-color: #fff;
        border: 1px solid #e7eaef;
        border-radius: 2px;
        color: #2f313c;
        height: 36px;
        padding: 0 10px;
        margin-right: 10px;
        &:hover{
          border-color: #d1495b;
          color: #d1495b;
        }
        .icon{
          padding-right: 5px;
        }
      }
      .scale{
        background-color: #fff;
        border: 1px solid #d1495b;
        color: #d1495b;
      }
      .toggle{
        cursor:pointer;
        display: flex;
        align-items: center;
        text-align: center;
        background-color: #fff;
        border: 1px solid #e7eaef;
        border-radius: 2px;
        color: #2f313c;
        height: 36px;
        padding: 0 10px;
        .icon{
          padding-right: 5px;
        }
        &:hover{
          background-color: #fff;
          border-color: #2f313c;
          color: #2f313c;
        }
      }
    }
    .rotate{
      cursor:pointer;
      display: flex;
      align-items: center;
      text-align: center;
      background-color: #fff;
      border: 1px solid #e7eaef;
      border-radius: 2px;
      color: #2f313c;
      height: 36px;
      padding: 0 10px;
      .icon{
        padding-right: 5px;
      }
      &:hover{
        background-color: #fff;
        border-color: #2f313c;
        color: #2f313c;
      }
    }
    .playerText{
      display: flex;
      align-items: center;
      border: 1px solid #e7eaef;
      border-radius: 2px;
      color: #2f313c;
      height: 36px;
      margin-left: 10px;
      padding: 0 10px;
      background-color: #fafafa;
      outline:none;
      font-size: 15px;
      font-weight: 400;
      &:focus{
        border-color: #45d695;
      }
    }
    .save{
      cursor:pointer;
      display: flex;
      height: 100%;
      width: 92px;
      justify-content: center;
      align-items: center;
      span{
        text-decoration: none;
        color: #fff;
        padding: 6px 14px;
        background-color: #45d695;
        border: 1px solid transparent;
        display: inline-block;
        border-radius: 500px;
        text-align: center;
        font-size: 14px;
        font-weight: 500;
        line-height: normal;
        &:hover{
          background-color: #2cc681;
          text-decoration: none;
        }
      }
    }
  }
</style>
