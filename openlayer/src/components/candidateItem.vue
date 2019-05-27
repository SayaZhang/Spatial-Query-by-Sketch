<template xmlns:>
  <div
    id="map-component"
    :class="[$options.name]"
    :style="{width: '100%', height: '90%'}"
  >
    <!-- map  -->
    <vl-map
      v-if="mapVisible"
      class="map"
      ref="map"
      :load-tiles-while-animating="true"
      :load-tiles-while-interacting="true"
      @click="clickCoordinate = $event.coordinate"
      @postcompose="onMapPostCompose"
      data-projection="EPSG:4326"
      @mounted="onMapMounted"
    >
      <!-- map view aka ol.View -->
      <vl-view ref="view" :center.sync="centerArray" :zoom.sync="zoom" :rotation.sync="rotation"></vl-view>

      <!-- base layers -->
      <vl-layer-tile
        v-for="layer in baseLayers"
        :key="layer.name"
        :id="layer.name"
        :visible="layer.visible"
      >
        <component :is="'vl-source-' + layer.name" v-bind="layer"></component>
      </vl-layer-tile>
      <!--// base layers -->
    </vl-map>
  </div>
</template>

<script>
import { kebabCase, camelCase } from "lodash";
import { findPointOnSurface, createStyle } from "vuelayers/lib/ol-ext";
import ScaleLine from "ol/control/ScaleLine";

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
    this.$refs.map.$map.getControls().extend([new ScaleLine()]);
  }
};

export default {
  name: "map-component",
  props: ["center", "title", "index"],
  methods,
  data() {
    return {
      zoom: 15,
      checkbox: false,
      rotation: 0,
      clickCoordinate: undefined,
      deviceCoordinate: undefined,
      mapPanel: {
        tab: "state"
      },
      panelOpen: true,
      mapVisible: true,
      cancel: {
        icon: "times",
        label: "Cancel"
      },
      mapGroup: [0,1],
      // base layers
      baseLayers: [
        {
          name: "osm",
          title: "OpenStreetMap",
          visible: true
        }
      ]
    };
  },
  computed: {
    centerArray: {
      get: function() {
        var arr = [];
        this.center.split(",").forEach(item => {
          arr.push(parseFloat(item));
        });
        console.log(arr)
        return arr;
      },
      set: function() {}
    }
  }
};
</script>

<style lang="sass">
  @import ~bulma/sass/utilities/_all
  
  .map-component
    position: relative

    .map
      height: 100%
      width: 100%

    .base-layers-panel
      position: absolute
      left: 50%
      bottom: 20px
      transform: translateX(-50%)

    .feature-popup
      position: absolute
      left: -50px
      bottom: 12px
      width: 20em
      max-width: none

      &:after, &:before
        top: 100%
        border: solid transparent
        content: ' '
        height: 0
        width: 0
        position: absolute
        pointer-events: none
      &:after
        border-top-color: white
        border-width: 10px
        left: 48px
        margin-left: -10px
      &:before
        border-top-color: #cccccc
        border-width: 11px
        left: 48px
        margin-left: -11px

      .card-content
        max-height: 20em
        overflow: auto

      .content
        word-break: break-all
</style>
