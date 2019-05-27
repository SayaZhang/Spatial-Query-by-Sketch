<template xmlns:>
  <b-modal :active.sync="downloading" width="80%">
    <div class="result-header">
      <span>Spatial-Query-by-Sketch Result</span>
      <div class="save">
        <span @click="save">Save</span>
      </div>
      <b-notification 
        type="is-success" 
        has-icon
        auto-close
        :active.sync="isSave"
        aria-close-label="Close notification">
        Save Success!
      </b-notification>
    </div>

    <div id="candidates">
      <div v-for="(candidate, index) in candidates" :key="index" class="candidate-panel">
        <div class="map-header">
          <span class="map-order">{{ index+1 }}.</span>
          <span class="map-title">{{ candidate[1] === undefined ? "" : candidate[1] }}</span>
          
          <span class="pick-title" 
            v-if="mapGroup.indexOf(index) > -1">
            {{mapGroup.indexOf(index) + 1}}
          </span>
          
          <b-checkbox
            v-model="mapGroup"
            type="is-success"
            class="map-check"
            :native-value="index"
          >YES</b-checkbox>         

        </div>  
          
        <candidateItem
          :center="candidate[0]"
          :title="candidate[1]"
          :index="index"
          class="candidate"
        />
        
      </div>
    </div>

    <div class="result-footer">
      <div class="text">
        <p>Not interest?</p>
        <p>不感兴趣？</p>
        <p>Please help us label the sketch!</p>
        <p>请帮助我们标记样本！</p>
        <p>Thanks!</p>
        <p>感谢您的贡献！</p>
      </div>

      <section class="extra-result">
        <b-field label="Semantic Location/地理语义">
          <b-input v-model="location" placeholder="Please enter a semantic location"></b-input>
        </b-field>

        <b-field label="Latitude & Longitude/经纬度">
          <b-input
            v-model="latlong"
            placeholder="Please enter latitude and longitude (like 116.29780630336761, 39.90493166365991)"
          ></b-input>
        </b-field>

        <!-- <candidateItem
          :center="latlong"
          :title="location"
          :index="99"
          class=""
          /> -->
      </section>
    </div>
  </b-modal>
</template>

<script>
import CandidateItem from "./candidateItem.vue";

export default {
  data() {
    return {
      mapGroup: [],
      latlong: "",//116.29780630336761, 39.90493166365991
      location: "",
      isSave: false
    };
  },
  components: {
    candidateItem: CandidateItem
  },
  computed: {
    candidates() {
      // var arr = [];
      // for (var i = 0; i < 12; i++) {
      //   arr.push(this.$store.state.candidates[0]);
      // }
      return this.$store.state.candidates;
    },
    downloading: {
      get() {
        return this.$store.state.downloading;
      },
      set() {
        this.$store.commit("changeDownloading", false);
      }
    },
    drawObjects() {
        return this.$store.state.drawObjects;
    }
  },
  methods: {
    save() {
      var y = [{}, {}];
      if (this.mapGroup.length > 0) {
        this.mapGroup.forEach((i, index) => {
          y[0][index] = this.candidates[i];
        });
      }

      if (this.latlong != "" || this.location != "") {
        y[1].latlong = this.latlong;
        y[1].location = this.location;
      }

      var that = this;

      this.$ajax({
        url: "http://159.226.172.85:5000/save",
        method: "post",
        data: {
          x: JSON.stringify(this.drawObjects),
          y: JSON.stringify(y),
        },
        dataType: 'JSON',
        transformRequest: [
          function(data) {
            let ret = "";
            for (let it in data) {
              ret +=
                encodeURIComponent(it) +
                "=" +
                encodeURIComponent(data[it]) +
                "&";
            }
            return ret;
          }
        ],
        // headers: {
        //   "Content-Type": "application/x-www-form-urlencoded"
        // }
      })
        .then(function() {
          that.isSave = true;
          console.log("Create!!");
        })
        .catch(function(error) {
          console.log(error);
        });
    }
  }
};
</script>

<style lang="sass">
  @import ~bulma/sass/utilities/_all

  #candidates
    margin: 5em 0 2% 0;
    background: #ffffff2b
    justify-content: center;
    display: flex;
    flex-wrap: wrap;
    color: #7a7a7a;
    .candidate-panel
      margin: 15px;
      width: 29%;
      background: #ffffffc2
    .map-header
      width: 100%
      padding: 5px;
      .map-title
        padding: 8px 0 0 0
      .map-check
        float: right
      span.pick-title
        float: right
        padding: 0 6px
        background: orange
        border-radius: 50%
        color: #fff
        margin-left: 5px
        line-height: 1.2em
  
  .result-header 
    display: flex
    position: fixed
    height: 3em
    line-height: 2em
    background: #fffffff0
    width: 80%
    color: #3F51B5
    font-weight: bold
    z-index: 100
    padding: 0.5em 1em
    article.notification.is-success
      height: 80px
      right: 0
      top: 4em
      position: absolute
    .save
        height: 2em
        float: right
        cursor:pointer
        display: flex
        justify-content: center
        align-items: center
        span
        text-decoration: none
        color: #fff
        padding: 6px 14px
        background-color: #45d695
        border: 1px solid transparent
        display: inline-block
        border-radius: 500px
        text-align: center
        font-size: 14px
        font-weight: 500
        line-height: normal
        line-height: normal
        position: absolute
        right: 1em
        &:hover
            background-color: #2cc681
            text-decoration: none
    
  .result-footer
    display: flex
    height: 15em
    line-height: 2em
    background: #ffffff2b
    width: 100%
    color: #fff
    z-index: 100
    padding: 0.25em 0.5em
    margin-bottom: 2em
    .text
      justify-content: center
      display: flex
      flex-direction: column
      margin-left: 1em
    .extra-result
      justify-content: center
      display: flex
      flex-direction: column
      flex: 1
      margin: 2em 5em
      label
        color: #fff
</style>