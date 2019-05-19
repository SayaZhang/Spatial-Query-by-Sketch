<template xmlns:>
  <div id="app" :class="[$options.name]">
    <!-- map  -->
    <vl-map v-if="mapVisible" class="map" ref="map" 
      :load-tiles-while-animating="true" 
      :load-tiles-while-interacting="true"
      @click="clickCoordinate = $event.coordinate"
      @postcompose="onMapPostCompose"
      data-projection="EPSG:4326" 
      @mounted="onMapMounted">
      
      <!-- map view aka ol.View -->
      <vl-view ref="view" :center.sync="center" :zoom.sync="zoom" :rotation.sync="rotation"></vl-view>

      <!-- geolocation -->
      <vl-geoloc @update:position="onUpdatePosition">
        <template slot-scope="geoloc">
          <vl-feature v-if="geoloc.position" id="position-feature">
            <vl-geom-point :coordinates="geoloc.position"></vl-geom-point>
            <vl-style-box>
              <vl-style-icon src="./assets/marker.png" :scale="0.4" :anchor="[0.5, 1]"></vl-style-icon>
            </vl-style-box>
          </vl-feature>
        </template>
      </vl-geoloc>
      <!--// geolocation -->

      <!-- circle geom -->
      <vl-feature id="circle">
        <vl-geom-circle :radius="1000000" :coordinates="[0, 30]"></vl-geom-circle>
      </vl-feature>
      <!--// circle geom -->

      <!-- base layers -->
      <vl-layer-tile v-for="layer in baseLayers" :key="layer.name" :id="layer.name" :visible="layer.visible">
        <component :is="'vl-source-' + layer.name" v-bind="layer"></component>
      </vl-layer-tile>
      <!--// base layers -->

    </vl-map>
    
    <!-- search map -->
    <vl-map class="search-panel" ref="search_map"
      :load-tiles-while-animating="true" 
      :load-tiles-while-interacting="true"
      @postcompose="onMapPostCompose"
      @mounted="onMapMounted">
      <!-- map view aka ol.View -->
      <vl-view :projection="searchMap.projection" :zoom.sync="searchMap.zoom" :center.sync="searchMap.center" :rotation.sync="searchMap.rotation"></vl-view>
      
      <vl-layer-image id="xkcd">
        <vl-source-image-static 
        :url="searchMap.imgUrl" 
        :size="searchMap.imgSize" 
        :extent="searchMap.extent" 
        :projection="searchMap.projection">
        </vl-source-image-static>
      </vl-layer-image>

      <!-- draw components -->
      <vl-layer-vector id="draw-pane" v-if="mapPanel.tab === 'draw'">
        <vl-source-vector ident="draw-target" :features.sync="drawnFeatures"></vl-source-vector>
      </vl-layer-vector>

      <vl-interaction-draw v-if="mapPanel.tab === 'draw' && drawType" source="draw-target" :type="drawType"></vl-interaction-draw>
      <vl-interaction-modify v-if="mapPanel.tab === 'draw'" source="draw-target"></vl-interaction-modify>
      <vl-interaction-snap v-if="mapPanel.tab === 'draw'" source="draw-target" :priority="10"></vl-interaction-snap>
      <!--// draw components -->

      <vl-layer-vector>
        <vl-source-vector>
          <!-- <vl-feature>
            <vl-geom-circle :coordinates="[0, 0]" :radius="40"></vl-geom-circle>
            <vl-style-box>
              <vl-style-stroke color="blue"></vl-style-stroke>
              <vl-style-fill color="rgba(255,255,255,0.5)"></vl-style-fill>
              <vl-style-text text="I'm circle"></vl-style-text>
            </vl-style-box>
          </vl-feature> -->
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

    <!-- map panel, controls -->
    <div class="map-panel">
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
          <a @click="showMapPanelTab('state')" :class="mapPanelTabClasses('state')">State</a>
          <a @click="showMapPanelTab('draw')" :class="mapPanelTabClasses('draw')">Draw</a>
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
            <tr>
              <th>Device coordinate</th>
              <td>{{ deviceCoordinate }}</td>
            </tr>
          </table>
        </div>

        <div class="panel-block draw-panel" v-show="mapPanel.tab === 'draw'">
          <div class="buttons">
            <button v-for="control in drawControls" :key="control.type || -1" @click="drawType = control.type"
                    :class="drawType && drawType === control.type ? 'is-info' : ''" class="button" >
              <b-icon :icon="control.icon"></b-icon>
              <span>{{ control.label }}</span>
            </button>
            <button class="button" @click="onMapCancel">
              <b-icon :icon="cancel.icon"></b-icon>
              <span>{{ cancel.label }}</span>
            </button>
          </div>
        </div>
      </b-collapse>
    </div>
    <!--// map panel, controls -->

    <!-- base layers switch -->
    <div class="base-layers-panel">
      <div class="buttons has-addons">
        <button class="button is-light" v-for="layer in baseLayers"
                :key="layer.name" :class="{ 'is-info': layer.visible }"
                @click="showBaseLayer(layer.name)">
          {{ layer.title }}
        </button>
        <button class="button is-light" @click="mapVisible = !mapVisible">
          {{ mapVisible ? 'Hide map' : 'Show map' }}
        </button>
      </div>
    </div>

  </div>
</template>

<script>
  import { kebabCase, camelCase } from 'lodash'
  import { createProj, addProj, findPointOnSurface, createStyle } from 'vuelayers/lib/ol-ext'
  import ScaleLine from 'ol/control/ScaleLine'
  import FullScreen from 'ol/control/FullScreen'
  import OverviewMap from 'ol/control/OverviewMap'
  import ZoomSlider from 'ol/control/ZoomSlider'

  // Custom projection for static Image layer
  let x = 40
  let y = 40
  let imageExtent = [0,0,x,y]
  let customProj = createProj({
    code: 'xkcd-image',
    units: 'pixels',
    extent: imageExtent,
  })
  addProj(customProj)

  const easeInOut = t => 1 - Math.pow(1 - t, 3)

  const methods = {
    camelCase,
    pointOnSurface: findPointOnSurface,
    geometryTypeToCmpName (type) {
      return 'vl-geom-' + kebabCase(type)
    },
    selectFilter (feature) {
      return ['position-feature'].indexOf(feature.getId()) === -1
    },
    onUpdatePosition (coordinate) {
      this.deviceCoordinate = coordinate
    },
    onMapPostCompose ({ vectorContext, frameState }) {
      if (!this.$refs.marker || !this.$refs.marker.$feature) return

      const feature = this.$refs.marker.$feature
      if (!feature.getGeometry() || !feature.getStyle()) return

      const flashGeom = feature.getGeometry().clone()
      const duration = feature.get('duration')
      const elapsed = frameState.time - feature.get('start')
      const elapsedRatio = elapsed / duration
      const radius = easeInOut(elapsedRatio) * 35 + 5
      const opacity = easeInOut(1 - elapsedRatio)
      const fillOpacity = easeInOut(0.5 - elapsedRatio)

      vectorContext.setStyle(createStyle({
        imageRadius: radius,
        fillColor: [119, 170, 203, fillOpacity],
        strokeColor: [119, 170, 203, opacity],
        strokeWidth: 2 + opacity,
      }))

      vectorContext.drawGeometry(flashGeom)
      vectorContext.setStyle(feature.getStyle()(feature)[0])
      vectorContext.drawGeometry(feature.getGeometry())

      if (elapsed > duration) {
        feature.set('start', Date.now())
      }

      this.$refs.map.render()
    },
    onMapCancel () {
      this.drawnFeatures.pop()
      //console.log(this.$refs.map)
      this.$refs.search_map.render()
    },
    onMapMounted () {
      // now ol.Map instance is ready and we can work with it directly
      this.$refs.map.$map.getControls().extend([
        new ScaleLine(),
        new FullScreen(),
        new OverviewMap({
          collapsed: false,
          collapsible: true,
        }),
        new ZoomSlider(),
      ])
    },
    // base layers
    showBaseLayer (name) {
      let layer = this.baseLayers.find(layer => layer.visible)
      if (layer != null) {
        layer.visible = false
      }

      layer = this.baseLayers.find(layer => layer.name === name)
      if (layer != null) {
        layer.visible = true
      }
    },
    // map panel
    mapPanelTabClasses (tab) {
      return {
        'is-active': this.mapPanel.tab === tab,
      }
    },
    showMapPanelLayer (layer) {
      layer.visible = !layer.visible
    },
    showMapPanelTab (tab) {
      this.mapPanel.tab = tab
      if (tab !== 'draw') {
        this.drawType = undefined
      }
    },
  }

  export default {
    name: 'vld-demo-app',
    methods,
    data () {
      return {
        center: [116.29780630336761, 39.90493166365991],
        zoom: 12,
        rotation: 0,
        clickCoordinate: undefined,
        deviceCoordinate: undefined,
        mapPanel: {
          tab: 'state',
        },
        searchMap: {
          zoom: 0.63,
          maxZoom: 2,
          minZoom: 0.63,
          center: [x / 2, y / 2],
          rotation: 0,
          extent: imageExtent,
          projection: customProj.getCode(),
          imgUrl: '../src/assets/grid.bmp',
          imgSize: [x, y],
        },
        panelOpen: true,
        mapVisible: true,
        drawControls: [
          {
            type: 'point',
            label: 'Draw Point',
            icon: 'map-marker',
          },
          {
            type: 'line-string',
            label: 'Draw LineString',
            icon: 'minus',
          },
          {
            type: 'polygon',
            label: 'Draw Polygon',
            icon: 'square-o',
          },
          {
            type: 'circle',
            label: 'Draw Circle',
            icon: 'circle-thin',
          },
          {
            type: undefined,
            label: 'Stop drawing',
            icon: 'times',
          },
        ],
        drawType: undefined,
        drawnFeatures: [],
        cancel: {
          icon: 'times',
          label: 'Cancel'
        },
        // base layers
        baseLayers: [
          {
            name: 'osm',
            title: 'OpenStreetMap',
            visible: true,
          },
          {
            name: 'sputnik',
            title: 'Sputnik Maps',
            visible: false,
          },
          {
            name: 'bingmaps',
            title: 'Bing Maps',
            apiKey: 'ArbsA9NX-AZmebC6VyXAnDqjXk6mo2wGCmeYM8EwyDaxKfQhUYyk0jtx6hX5fpMn',
            imagerySet: 'CanvasGray',
            visible: false,
          },
        ],
      }
    },
    updated: function () {
      console.log(this.drawnFeatures)
      console.log()
    }
  }
</script>

<style lang="sass">
  @import ~bulma/sass/utilities/_all

  html, body, #app
    width: 100%
    height: 100%
    margin: 0
    padding: 0

  .vld-demo-app
    position: relative

    .map
      height: 100%
      width: 100%

    .search-panel
      width: 25em
      height: 25em
      position: absolute
      left: 3em
      top: 360px
      background: #fafafae0

    .map-panel
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
            .button
              display: block
              flex: 1 1 100%

      +tablet()
        position: absolute
        top: 0
        left: 3em
        max-height: 500px
        width: 25em

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
