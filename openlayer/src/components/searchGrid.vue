<template xmlns:>
  <div id="search-component" :class="[$options.name]">
    <!-- head panel, controls -->
    <div class="head-panel">
      <b-collapse class="panel box is-paddingless" :open.sync="panelOpen">
        <div slot="trigger" class="panel-heading">
          <div class="columns">
            <div class="column is-11">
              <strong>Map panel</strong>
            </div>
            <div class="column">
              <b-icon :icon="panelOpen ? 'chevron-up' : 'chevron-down'" size="is-small"></b-icon>
            </div>
          </div>
        </div>
        <p class="panel-tabs">
          <a @click="showMapPanelTab('draw')" :class="mapPanelTabClasses('draw')">样本收集/Collect</a>
          <a @click="showMapPanelTab('search')" :class="mapPanelTabClasses('search')">草图搜索/Search</a>
          <a @click="showMapPanelTab('state')" :class="mapPanelTabClasses('state')">当前设置/State</a>
        </p>

        <div class="panel-block" v-show="mapPanel.tab === 'state'">
          <table class="table is-fullwidth">
            <tr>
              <th>Map center</th>
              <td>{{ center }}</td>
            </tr>
            <tr>
              <th>Map zoom</th>
              <td>{{ zoom }}</td>
            </tr>
            <tr>
              <th>Map rotation</th>
              <td>{{ rotation }}</td>
            </tr>
          </table>
        </div>

        <div class="panel-block draw-panel" v-show="mapPanel.tab === 'draw'">
          <div class="buttons">
            <div class="objects block">
              <b-radio
                size="is-small"
                v-for="object in drawObjects"
                v-model="drawObjectType"
                :native-value="object.type"
                :key="object.type || -1"
              >
                <span>{{ object.type }}</span>
              </b-radio>
            </div>

            <div class="input-pos">
              <b-input
                v-model="drawPosition"
                size="is-small"
                placeholder="请输入样本经纬度"
                icon="map-marker"
                rounded
              ></b-input>
            </div>

            <div class="shape">
              <button
                v-for="control in drawControls"
                :key="control.type || -1"
                @click="drawType = control.type"
                :class="drawType && drawType === control.type ? 'is-info' : ''"
                class="button"
              >
                <b-icon :icon="control.icon"></b-icon>
                <span>{{ control.label }}</span>
              </button>
            </div>

            <div class="setting">
              <button class="button" @click="drawType = setting.stop.type">
                <b-icon :icon="setting.stop.icon"></b-icon>
                <span>{{ setting.stop.label }}</span>
              </button>

              <button class="button" @click="drawUndo">
                <b-icon :icon="setting.undo.icon"></b-icon>
                <span>{{ setting.undo.label }}</span>
              </button>

              <button class="button" @click="drawComplete">
                <b-icon :icon="setting.complete.icon"></b-icon>
                <span>{{ setting.complete.label }}</span>
              </button>
              <b-notification
                type="is-success"
                has-icon
                auto-close
                :active.sync="isCreate"
                aria-close-label="Close notification"
              >Create!!<br>Thanks!!</b-notification>
              <b-notification
                type="is-warning"
                has-icon
                :active.sync="isWarning"
                aria-close-label="Close notification"
              >Empty<br>Position!!</b-notification>
            </div>

            <!-- <span id="grid-tip">* Each record are mapped into a 40*40 grid with cells of size 10*10 meters</span> -->
          </div>
        </div>

        <div class="panel-block draw-panel" v-show="mapPanel.tab === 'search'">
          <div class="buttons">
            <div class="objects block">
              <b-radio
                size="is-small"
                v-for="object in drawObjects"
                v-model="drawObjectType"
                :native-value="object.type"
                :key="object.type || -1"
              >
                <span>{{ object.type }}</span>
              </b-radio>
            </div>

            <div class="shape">
              <button
                v-for="control in drawControls"
                :key="control.type || -1"
                @click="drawType = control.type"
                :class="drawType && drawType === control.type ? 'is-info' : ''"
                class="button"
              >
                <b-icon :icon="control.icon"></b-icon>
                <span>{{ control.label }}</span>
              </button>
            </div>

            <div class="setting">
              <button class="button" @click="drawType = setting.stop.type">
                <b-icon :icon="setting.stop.icon"></b-icon>
                <span>{{ setting.stop.label }}</span>
              </button>

              <button class="button" @click="drawUndo">
                <b-icon :icon="setting.undo.icon"></b-icon>
                <span>{{ setting.undo.label }}</span>
              </button>

              <button class="button" @click="search">
                <b-icon :icon="setting.complete.icon"></b-icon>
                <span>{{ setting.complete.label }}</span>
              </button>
            </div>
          </div>
        </div>
      </b-collapse>
    </div>

    <!-- search map -->
    <vl-map
      class="search-panel"
      ref="map"
      :load-tiles-while-animating="true"
      :load-tiles-while-interacting="true"
      :controls="false"
      @postcompose="onMapPostCompose"
      @mounted="onMapMounted"
    >
      <!-- map view aka ol.View -->
      <vl-view
        :projection="projection"
        :zoom.sync="zoom"
        :center.sync="center"
        :rotation.sync="rotation"
        :max-zoom="zoom"
        :min-zoom="zoom"
      ></vl-view>

      <vl-layer-image id="xkcd">
        <vl-source-image-static
          :url="imgUrl"
          :size="imgSize"
          :extent="extent"
          :projection="projection"
        ></vl-source-image-static>
      </vl-layer-image>

      <!-- draw components -->
      <vl-layer-vector id="draw-pane">
        <vl-source-vector ident="draw-target" :features.sync="drawnFeatures" ref="feature"></vl-source-vector>
        <vl-style-box>
          <vl-style-icon :scale="0.2" src="../assets/marker.png"></vl-style-icon>
          <vl-style-stroke :width="3" color="orange"></vl-style-stroke>
          <vl-style-fill color="rgba(255,255,255,0.25)"></vl-style-fill>
        </vl-style-box>
      </vl-layer-vector>

      <vl-interaction-draw
        v-if="mapPanel.tab === 'draw' && drawType"
        source="draw-target"
        :type="drawType"
      ></vl-interaction-draw>
      <vl-interaction-modify v-if="mapPanel.tab === 'draw'" source="draw-target"></vl-interaction-modify>
      <vl-interaction-snap v-if="mapPanel.tab === 'draw'" source="draw-target" :priority="10"></vl-interaction-snap>

      <!-- border -->
      <vl-layer-vector>
        <vl-source-vector>
          <vl-feature>
            <vl-geom-polygon :coordinates="[[[0, 0], [40, 0], [40, 40], [0, 40]]]"></vl-geom-polygon>
            <vl-style-box>
              <vl-style-stroke color="orange"></vl-style-stroke>
              <vl-style-fill color="rgba(255,255,255,0.5)"></vl-style-fill>
            </vl-style-box>
          </vl-feature>
        </vl-source-vector>
      </vl-layer-vector>
    </vl-map>
  </div>
</template>

<script>
import { kebabCase, camelCase } from "lodash";
import {
  createProj,
  addProj,
  findPointOnSurface,
  createStyle
} from "vuelayers/lib/ol-ext";

// Custom projection for static Image layer
let x = 40;
let y = 40;
let imageExtent = [0, 0, x, y];
let customProj = createProj({
  code: "xkcd-image",
  units: "pixels",
  extent: imageExtent
});
addProj(customProj);

const easeInOut = t => 1 - Math.pow(1 - t, 3);

const methods = {
  camelCase,
  pointOnSurface: findPointOnSurface,
  geometryTypeToCmpName(type) {
    return "vl-geom-" + kebabCase(type);
  },
  selectFilter(feature) {
    return ["position-feature"].indexOf(feature.getId()) === -1;
  },
  onUpdatePosition(coordinate) {
    this.deviceCoordinate = coordinate;
  },
  onMapPostCompose({ vectorContext, frameState }) {
    if (!this.$refs.marker || !this.$refs.marker.$feature) return;

    const feature = this.$refs.marker.$feature;
    if (!feature.getGeometry() || !feature.getStyle()) return;

    const flashGeom = feature.getGeometry().clone();
    const duration = feature.get("duration");
    const elapsed = frameState.time - feature.get("start");
    const elapsedRatio = elapsed / duration;
    const radius = easeInOut(elapsedRatio) * 35 + 5;
    const opacity = easeInOut(1 - elapsedRatio);
    const fillOpacity = easeInOut(0.5 - elapsedRatio);

    vectorContext.setStyle(
      createStyle({
        imageRadius: radius,
        fillColor: [119, 170, 203, fillOpacity],
        strokeColor: [119, 170, 203, opacity],
        strokeWidth: 2 + opacity
      })
    );

    vectorContext.drawGeometry(flashGeom);
    vectorContext.setStyle(feature.getStyle()(feature)[0]);
    vectorContext.drawGeometry(feature.getGeometry());

    if (elapsed > duration) {
      feature.set("start", Date.now());
    }

    this.$refs.map.render();
  },
  onMapMounted() {
    // now ol.Map instance is ready and we can work with it directly
    this.$refs.map.$map.getControls().extend([]);
  },
  // map panel
  mapPanelTabClasses(tab) {
    return {
      "is-active": this.mapPanel.tab === tab
    };
  },
  showMapPanelLayer(layer) {
    layer.visible = !layer.visible;
  },
  showMapPanelTab(tab) {
    this.mapPanel.tab = tab;
    if (tab !== "draw") {
      this.drawType = undefined;
    }
  },
  // draw
  changeObject(type) {
    this.drawObjectType = type;
    console.log(this.drawObjectType);
  },
  drawUndo() {
    this.drawnFeatures.pop();
    this.drawResults.pop();
    this.$refs.feature.remount();
  },
  drawComplete() {
    if(this.drawPosition == ''){
      this.isWarning = true;
      return;
    }

    var that = this;
    this.$ajax({
      url: "http://localhost:5000/save",
      method: "post",
      data: {
        sketch: JSON.stringify(this.drawResults),
        position: this.drawPosition
      },
      transformRequest: [
        function(data) {
          let ret = "";
          for (let it in data) {
            ret +=
              encodeURIComponent(it) + "=" + encodeURIComponent(data[it]) + "&";
          }
          return ret;
        }
      ],
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      }
    })
      .then(function() {
        that.isCreate = true;
        console.log("Create!!");
      })
      .catch(function(error) {
        console.log(error);
      });
  },
  search() {}
};

export default {
  name: "search-component",
  methods,
  data() {
    return {
      zoom: 0.63,
      maxZoom: 2,
      minZoom: 0.63,
      center: [x / 2, y / 2],
      rotation: 0,
      extent: imageExtent,
      projection: customProj.getCode(),
      imgUrl: "../src/assets/grid.bmp",
      imgSize: [x, y],
      panelOpen: true,
      mapVisible: true,
      mapPanel: {
        tab: "draw"
      },
      drawType: undefined,
      drawObjectType: "学校及科教服务/School",
      drawnFeatures: [],
      drawResults: [],
      drawPosition: "",
      isCreate: false,
      isWarning: false,
      drawControls: [
        {
          type: "point",
          label: "Point",
          icon: "map-marker"
        },
        {
          type: "line-string",
          label: "LineString",
          icon: "minus"
        },
        {
          type: "polygon",
          label: "Polygon",
          icon: "square-o"
        },
        {
          type: "circle",
          label: "Circle",
          icon: "circle-thin"
        }
      ],
      setting: {
        stop: {
          type: undefined,
          label: "暂停绘制/Stop",
          icon: "times"
        },
        undo: {
          icon: "undo",
          label: "撤销/Undo"
        },
        complete: {
          icon: "check",
          label: "完成/Complete"
        }
      },
      drawObjects: [
        {
          type: "学校及科教服务/School",
          label: "School",
          icon: "map-marker"
        },
        {
          type: "政府机构及公共设施/Institute",
          label: "Institute",
          icon: "map-marker"
        },
        // {
        //   type: "Building",
        //   label: "Building",
        //   icon: "map-marker"
        // },
        {
          type: "住宿服务/Hotel",
          label: "Hotel",
          icon: "map-marker"
        },
        // {
        //   type: "Landuse",
        //   label: "Draw Point",
        //   icon: "map-marker"
        // },
        {
          type: "道路及相关设施/Road",
          label: "Draw LineString",
          icon: "map-marker"
        },
        {
          type: "植被及自然景观/Natural",
          label: "Draw Polygon",
          icon: "map-marker"
        },
        {
          type: "风景名胜/Scene",
          label: "Draw Polygon",
          icon: "map-marker"
        },
        {
          type: "餐饮服务/Repast",
          label: "Draw Circle",
          icon: "map-marker"
        },
        {
          type: "住宅区/Residence",
          label: "Draw Point",
          icon: "map-marker"
        },
        {
          type: "河流/River",
          label: "Draw LineString",
          icon: "map-marker"
        },
        {
          type: "购物服务/Shop",
          label: "Draw Polygon",
          icon: "map-marker"
        },
        {
          type: "写字楼等商业用地/Company",
          label: "Draw Circle",
          icon: "map-marker"
        },
        {
          type: "医院及其他医疗场所/Hospital",
          label: "Draw Point",
          icon: "map-marker"
        },
        {
          type: "生活休闲服务/Service",
          label: "Draw LineString",
          icon: "map-marker"
        }
      ]
    };
  },
  updated: function() {
    //console.log(this.drawObjectType);
  },
  watch: {
    drawnFeatures: {
      handler(newValue, oldValue) {
        var count = newValue.length;
        if (
          this.drawObjectType !== undefined &&
          count != 0 &&
          newValue[count - 1].properties == undefined &&
          count > oldValue.length
        ) {
          newValue[count - 1].properties = this.drawObjectType;
          this.drawResults.push(newValue[count - 1]);
        }
      }
    }
  }
};
</script>

<style lang="sass">
  @import ~bulma/sass/utilities/_all

  .search-component
    position: absolute
    top: 5%
    left: 5%

    .head-panel
      padding: 0
      
      .panel
        background: #fafafae0

      .panel-heading
        box-shadow: 0 .25em .5em transparentize($dark, 0.8)

      .panel-content
        background: $white
        box-shadow: 0 .25em .5em transparentize($dark, 0.8)

      .panel-block
        &.draw-panel
          .buttons
            flex-direction: column
            text-align: center
            .button
              flex: 1 1 100%
              font-size: 0.8em
              background-color: #ffffff66
              border-color: #dbdbdb87
            .button.is-info
              border-color: #167df082
              color: #2196F3
            .setting
              display: flex
              background: #673ab721
              padding: 5px
              width: 23em
              .button
                margin-bottom: 0
              .notification
                position: absolute
                right: -200px
                top: 300px
                font-weight: bold

            .shape
              background: #2196f31f
              padding: 5px
              margin-bottom: 10px
              width: 23em
              .button
                margin-bottom: 0

            .objects
              padding: 5px
              text-align: left
              max-width: 96%
              margin-bottom: 10px
              background: #d6bcbc21
              .label
                background: none
                border: 0
                font-weight: bold
                color: #3F51B5
            
            .input-pos
              width: 96%
              margin-bottom: 10px
          
            span#grid-tip
                font-size: 10px
                font-family: -webkit-pictograph
                margin: 8px 0 -8px 0
          
          .buttons:last-child
            margin-bottom: 0

      +tablet()
        top: 0
        max-height: 500px
        width: 25em

    .search-panel
        width: 25em
        height: 25em    
        margin-top: 20px
        background: #fafafae0
        span
          font-size: 0.8em
          background: #fff
          padding: 3px
          z-index: 9
          position: absolute
          margin-top: 10px
          margin-left: 10px
          border: 2px solid #df9c21
    
</style>
