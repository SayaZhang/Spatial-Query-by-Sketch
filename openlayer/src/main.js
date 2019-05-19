import '@babel/polyfill'
import Vue from 'vue'
import Buefy from 'buefy'
import 'buefy/dist/buefy.css'
// import VueLayers modules
import VueLayers from 'vuelayers'
import axios from 'axios'
//import VueAxios from 'vue-axios'
// import VueLayers styles
import 'vuelayers/lib/style.css'
import App from './App.vue'
import store from './store/store'

Vue.config.productionTip = false
Vue.use(Buefy, {
  defaultIconPack: 'fa',
})
// register all VueLayers components
Vue.use(VueLayers, {
  // global data projection, see https://vuelayers.github.io/#/quickstart?id=global-data-projection
  // dataProjection: 'EPSG:4326',
})
Vue.prototype.$ajax = axios
Vue.prototype.HOME = '/api'
Vue.config.productionTip = false

/* eslint-disable no-new */
new Vue({
  el: '#app',
  store,
  render: h => h(App)
})
